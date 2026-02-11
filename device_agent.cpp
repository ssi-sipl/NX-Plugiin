// Optimized version with asynchronous processing to eliminate lag

#include "device_agent.h"
#include <chrono>
#include <thread>
#include <queue>
#include <condition_variable>
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <sstream>

#include <nx/kit/debug.h>
#include <nx/sdk/analytics/helpers/event_metadata.h>
#include <nx/sdk/analytics/helpers/event_metadata_packet.h>
#include <nx/sdk/helpers/attribute.h>
#include <nx/sdk/analytics/i_uncompressed_video_frame.h>
#include <nx/sdk/analytics/helpers/object_metadata.h>
#include <nx/sdk/analytics/helpers/object_metadata_packet.h>
#include <nx/sdk/analytics/helpers/object_track_best_shot_packet.h>
#include <nx/sdk/analytics/i_motion_metadata_packet.h>

#include "device_agent_manifest.h"
#include "object_attributes.h"
#include "../utils.h"
#include "stub_analytics_plugin_object_detection_ini.h"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>

namespace nx {
namespace vms_server_plugins {
namespace analytics {
namespace stub {
namespace object_detection {

using namespace nx::sdk;
using namespace nx::sdk::analytics;
using Uuid = nx::sdk::Uuid;

// Adaptive detection parameters
static constexpr float kFpsChangeFpsDifferenceThreshold = 0.10F;
static constexpr int64_t kFpsChangeTimestampUsDifferenceThreshold = 5000000;

static const std::string kDetectionsBasePath = "/opt/networkoptix/detections";


// Object tracking parameters
static constexpr float kIoUThreshold = 0.4f;
static constexpr int kMaxFramesLost = 15;

const std::string DeviceAgent::kTimeShiftSetting = "timestampShiftMs";
const std::string DeviceAgent::kSendAttributesSetting = "sendAttributes";
const std::string DeviceAgent::kObjectTypeGenerationSettingPrefix = "objectTypeIdToGenerate.";
const std::string DeviceAgent::kModelPathSetting = "modelPath";
const std::string DeviceAgent::kClassesPathSetting = "classesPath";
const std::string DeviceAgent::kConfidenceThresholdSetting = "confidenceThreshold";
const std::string DeviceAgent::kNmsThresholdSetting = "nmsThreshold";
const std::string DeviceAgent::kDetectionRegionSetting = "detectionRegion";
const std::string DeviceAgent::kEnableDetectionRegionSetting = "enableDetectionRegion";  // ADD THIS
const std::string DeviceAgent::kUseGpuSetting = "useGpu";


// Tracking structure
struct TrackedObject {
    Uuid trackId;
    cv::Rect lastBox;
    std::string objectType;
    int framesLost;
    int64_t lastSeenTimestamp;
    cv::Mat bestShotImage;  // Store best shot for notifications
    float bestShotQuality;
    int64_t lastEventTimestampUs = 0;
    bool bestShotSent = false; 
      
    TrackedObject(Uuid id, cv::Rect box, std::string type, int64_t timestamp)
        : trackId(id), lastBox(box), objectType(type), framesLost(0), 
          lastSeenTimestamp(timestamp), bestShotQuality(0.0f) {}
};

constexpr int64_t kEventIntervalUs = 1'000'000; // 1 second


// Frame queue structure for async processing
struct FrameData {
    cv::Mat frame;
    int64_t timestampUs;
    int frameIndex;
    bool hasMotion;
};

DeviceAgent::DeviceAgent(const nx::sdk::IDeviceInfo* deviceInfo):
    ConsumingDeviceAgent(deviceInfo, /*enableOutput*/ true),
    m_fps(25.0f),
    m_detectionFrequencyFps(5),
    m_objectDetectionPeriod(5),
    m_exponentialBackoffPeriod(5.0f),
    m_maxExponentialBackoffPeriod(300.0f),
    m_isExponentialBackoffActive(false),
    m_fpsChangeTimestampUs(0),
    m_lastTrackCleanupFrame(0),
    m_processingThreadRunning(false),
    m_terminateThread(false),
    // m_modelPath("/opt/networkoptix/mediaserver/bin/plugins/stub_analytics_plugin/models/yolov5s.onnx"),
    m_modelPath(""),
    m_classesPath(""),
    // m_classesPath("/opt/networkoptix/mediaserver/bin/plugins/stub_analytics_plugin/models/classes.txt"),
    m_confidenceThreshold(0.75f),
    m_nmsThreshold(0.45f),
    m_useDetectionRegion(false),
    m_useGpu(false)
{
    NX_PRINT << "DeviceAgent constructor called";

    m_cameraId = deviceInfo->id(); // const char* → std::string (OK)
    NX_PRINT << "Camera ID: " << m_cameraId;

    loadModel();

    m_objectDetectionPeriod = calcObjectDetectionPeriod();
    m_exponentialBackoffPeriod = static_cast<float>(m_objectDetectionPeriod);
    
    // Start async processing thread
    m_processingThreadRunning = true;
    m_processingThread = std::thread(&DeviceAgent::processingThreadFunc, this);
}

DeviceAgent::~DeviceAgent()
{
    NX_PRINT << "DeviceAgent destructor called";
    
    // Stop processing thread
    {
        std::lock_guard<std::mutex> lock(m_queueMutex);
        m_terminateThread = true;
        m_queueCondition.notify_all();
    }
    
    if (m_processingThread.joinable())
        m_processingThread.join();
}

std::string DeviceAgent::manifestString() const
{
    return kDeviceAgentManifest;
}

static std::string normalizeClassName(std::string s)
{
    // trim whitespace
    s.erase(s.begin(), std::find_if(s.begin(), s.end(),
        [](unsigned char ch) { return !std::isspace(ch); }));

    s.erase(std::find_if(s.rbegin(), s.rend(),
        [](unsigned char ch) { return !std::isspace(ch); }).base(), s.end());

    // lowercase
    std::transform(s.begin(), s.end(), s.begin(),
        [](unsigned char c) { return std::tolower(c); });

    return s;
}

std::string DeviceAgent::mapCocoClassToNxType(const std::string& cocoClass)
{
    static std::map<std::string, std::string> mapping = {
        {"person", "nx.base.Person"},
        {"Person", "nx.base.Person"},
        {"car", "nx.base.Car"},
        {"Car", "nx.base.Car"},
        {"truck", "nx.base.Truck"},
        {"bus", "nx.base.Bus"},
        {"train", "nx.base.Train"},
        {"motorcycle", "nx.base.Bike"},
        {"bicycle", "nx.base.Bike"},
        {"boat", "nx.base.WaterTransport"},
        {"airplane", "nx.base.AirTransport"},
        {"bird", "nx.base.Bird"},
        {"cat", "nx.base.Cat"},
        {"dog", "nx.base.Dog"},
        {"horse", "nx.base.Animal"},
        {"sheep", "nx.base.Animal"},
        {"cow", "nx.base.Animal"},
        {"bear", "nx.base.Animal"},
        {"zebra", "nx.base.Animal"},
        {"giraffe", "nx.base.Animal"},
        {"elephant", "nx.base.Animal"},
        {"Elephant", "nx.base.Animal"},
        {"leopard", "nx.base.Animal"}
    };
    
    auto it = mapping.find(cocoClass);
    if (it != mapping.end())
        return it->second;
    
    return "nx.base.Unknown";
}

float DeviceAgent::calculateIoU(const cv::Rect& box1, const cv::Rect& box2)
{
    int x1 = std::max(box1.x, box2.x);
    int y1 = std::max(box1.y, box2.y);
    int x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    int y2 = std::min(box1.y + box1.height, box2.y + box2.height);
    
    int intersectionArea = std::max(0, x2 - x1) * std::max(0, y2 - y1);
    int box1Area = box1.width * box1.height;
    int box2Area = box2.width * box2.height;
    int unionArea = box1Area + box2Area - intersectionArea;
    
    return unionArea > 0 ? static_cast<float>(intersectionArea) / unionArea : 0.0f;
}

// Calculate image quality score for best shot selection
float DeviceAgent::calculateImageQuality(const cv::Mat& image, const cv::Rect& box)
{
    if (box.width <= 0 || box.height <= 0)
        return 0.0f;
    
    // Extract ROI
    cv::Rect safeBox(
        std::max(0, box.x),
        std::max(0, box.y),
        std::min(box.width, image.cols - box.x),
        std::min(box.height, image.rows - box.y)
    );
    
    if (safeBox.width <= 0 || safeBox.height <= 0)
        return 0.0f;
    
    cv::Mat roi = image(safeBox);
    
    // Calculate sharpness using Laplacian variance
    cv::Mat gray, laplacian;
    cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    cv::Laplacian(gray, laplacian, CV_64F);
    
    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacian, mean, stddev);
    float sharpness = stddev[0] * stddev[0];
    
    // Size factor (prefer larger detections)
    float sizeFactor = std::sqrt(static_cast<float>(box.width * box.height));
    
    // Center factor (prefer centered objects)
    float centerX = box.x + box.width / 2.0f;
    float centerY = box.y + box.height / 2.0f;
    float imgCenterX = image.cols / 2.0f;
    float imgCenterY = image.rows / 2.0f;
    float centerDistance = std::sqrt(
        std::pow(centerX - imgCenterX, 2) + std::pow(centerY - imgCenterY, 2));
    float maxDistance = std::sqrt(
        std::pow(imgCenterX, 2) + std::pow(imgCenterY, 2));
    float centerFactor = 1.0f - (centerDistance / maxDistance);
    
    // Combined quality score
    return sharpness * 0.5f + sizeFactor * 0.3f + centerFactor * 100.0f;
}

TrackMatchResult DeviceAgent::matchOrCreateTrack(
    const cv::Rect& box, 
    const std::string& objectType, 
    int64_t timestamp,
    const cv::Mat& frame,
    float confidence)
{
    float bestIoU = 0.0f;
    int bestTrackIdx = -1;
    
    for (size_t i = 0; i < m_trackedObjects.size(); ++i)
    {
        if (m_trackedObjects[i].objectType != objectType)
            continue;
            
        float iou = calculateIoU(box, m_trackedObjects[i].lastBox);
        if (iou > bestIoU && iou > kIoUThreshold)
        {
            bestIoU = iou;
            bestTrackIdx = i;
        }
    }
    
    if (bestTrackIdx >= 0)
    {
        auto& track = m_trackedObjects[bestTrackIdx];
        track.lastBox = box;
        track.framesLost = 0;
        track.lastSeenTimestamp = timestamp;
        
        // Update best shot if this frame is better quality
        float quality = calculateImageQuality(frame, box);
        if (quality > track.bestShotQuality)
        {
            // Extract and store cropped object image with safety checks
            cv::Rect safeBox(
                std::max(0, box.x),
                std::max(0, box.y),
                std::min(box.width, frame.cols - std::max(0, box.x)),
                std::min(box.height, frame.rows - std::max(0, box.y))
            );
            
            // Verify box is valid
            if (safeBox.width > 0 && safeBox.height > 0 &&
                safeBox.x >= 0 && safeBox.y >= 0 &&
                safeBox.x + safeBox.width <= frame.cols &&
                safeBox.y + safeBox.height <= frame.rows)
            {
                try {
                    track.bestShotImage = frame(safeBox).clone();
                    track.bestShotQuality = quality;
                    track.bestShotSent = false;
                    NX_PRINT << "Updated best shot for track (quality: " << quality 
                             << ", size: " << safeBox.width << "x" << safeBox.height << ")";
                } catch (const cv::Exception& e) {
                    NX_PRINT << "ERROR: Failed to crop best shot: " << e.what();
                }
            }
            else
            {
                NX_PRINT << "WARNING: Invalid crop box (" << safeBox.x << ", " << safeBox.y 
                         << ", " << safeBox.width << ", " << safeBox.height 
                         << ") for frame " << frame.cols << "x" << frame.rows;
            }
        }
        
        return { track.trackId, false };
}
    
    else
    {
        // Create new track
        Uuid newTrackId = nx::sdk::UuidHelper::randomUuid();
        m_trackedObjects.emplace_back(newTrackId, box, objectType, timestamp);
        
        auto& track = m_trackedObjects.back();
        track.bestShotSent = false;
        
        // Store initial best shot with safety checks
        cv::Rect safeBox(
            std::max(0, box.x),
            std::max(0, box.y),
            std::min(box.width, frame.cols - std::max(0, box.x)),
            std::min(box.height, frame.rows - std::max(0, box.y))
        );
        
        // Verify box is valid
        if (safeBox.width > 0 && safeBox.height > 0 &&
            safeBox.x >= 0 && safeBox.y >= 0 &&
            safeBox.x + safeBox.width <= frame.cols &&
            safeBox.y + safeBox.height <= frame.rows)
        {
            try {
                track.bestShotImage = frame(safeBox).clone();
                track.bestShotQuality = calculateImageQuality(frame, box);
                NX_PRINT << "Stored initial best shot (size: " << safeBox.width << "x" << safeBox.height << ")";
            } catch (const cv::Exception& e) {
                NX_PRINT << "ERROR: Failed to crop initial best shot: " << e.what();
            }
        }
        else
        {
            NX_PRINT << "WARNING: Invalid initial crop box (" << safeBox.x << ", " << safeBox.y 
                     << ", " << safeBox.width << ", " << safeBox.height 
                     << ") for frame " << frame.cols << "x" << frame.rows;
        }
        
        NX_PRINT << "New track created: " << objectType << " (total tracks: " << m_trackedObjects.size() << ")";
        return { newTrackId, true };
    }
}

void DeviceAgent::cleanupLostTracks()
{
    for (auto& track : m_trackedObjects)
        track.framesLost++;
    
    auto it = std::remove_if(m_trackedObjects.begin(), m_trackedObjects.end(),
        [](const TrackedObject& track) { return track.framesLost > kMaxFramesLost; });
    
    if (it != m_trackedObjects.end())
    {
        int removed = std::distance(it, m_trackedObjects.end());
        m_trackedObjects.erase(it, m_trackedObjects.end());
        NX_PRINT << "Removed " << removed << " lost tracks (remaining: " << m_trackedObjects.size() << ")";
    }
}

bool DeviceAgent::needToDetectObjects(const IUncompressedVideoFrame* videoFrame)
{
    if (m_frameIndex % m_objectDetectionPeriod != 0)
        return false;

    if (m_isExponentialBackoffActive)
    {
        if (m_frameIndex % static_cast<int>(m_exponentialBackoffPeriod) == 0)
            return true;
        return false;
    }

    const Ptr<IList<IMetadataPacket>> metadataPacketList = videoFrame->metadataList();
    if (!metadataPacketList)
        return true;

    const int metadataPacketCount = metadataPacketList->count();
    if (metadataPacketCount == 0)
        return true;

    for (int i = 0; i < metadataPacketCount; ++i)
    {
        const auto metadataPacket = metadataPacketList->at(i);
        const auto motionPacket = metadataPacket->queryInterface<IMotionMetadataPacket>();
        if (motionPacket && !motionPacket->isEmpty())
            return true;
    }

    return false;
}

void DeviceAgent::updateFpsAndDetectionPeriod(int64_t timestampUs)
{
    if (m_fpsChangeTimestampUs == 0)
    {
        m_fpsChangeTimestampUs = timestampUs;
        return;
    }

    int64_t timeDiff = timestampUs - m_fpsChangeTimestampUs;
    
    if (timeDiff > kFpsChangeTimestampUsDifferenceThreshold)
    {
        float newFps = static_cast<float>(m_frameIndex) / (timeDiff / 1000000.0f);
        
        if (std::fabs(newFps - m_fps) / m_fps > kFpsChangeFpsDifferenceThreshold)
        {
            NX_PRINT << "FPS changed from " << m_fps << " to " << newFps;
            m_fps = newFps;
            m_objectDetectionPeriod = calcObjectDetectionPeriod();
            m_exponentialBackoffPeriod = static_cast<float>(m_objectDetectionPeriod);
            
            m_frameIndex = 0;
            m_fpsChangeTimestampUs = timestampUs;
        }
    }
}

int DeviceAgent::calcObjectDetectionPeriod() const
{
    int period = std::max(static_cast<int>(m_fps / static_cast<float>(m_detectionFrequencyFps)), 1);
    NX_PRINT << "Detection period: " << period << " frames (FPS: " << m_fps << ", target: " << m_detectionFrequencyFps << "/sec)";
    return period;
}

// CRITICAL: Return immediately, queue frame for async processing
bool DeviceAgent::pushUncompressedVideoFrame(const IUncompressedVideoFrame* videoFrame)
{
    if (!videoFrame)
        return true;

    ++m_frameIndex;
    
    // Update FPS estimation periodically
    if (m_frameIndex % 100 == 0)
        updateFpsAndDetectionPeriod(videoFrame->timestampUs());

    // Check if we need to process this frame
    bool shouldProcess = needToDetectObjects(videoFrame);
    
    if (!shouldProcess)
    {
        // Still update tracking counters
        if (m_frameIndex % m_objectDetectionPeriod == 1)
        {
            std::lock_guard<std::mutex> lock(m_trackingMutex);
            for (auto& track : m_trackedObjects)
                track.framesLost++;
        }
        return true;
    }

    // Extract frame data
    int width = videoFrame->width();
    int height = videoFrame->height();

    auto mediaFrame = videoFrame->queryInterface<nx::sdk::analytics::IUncompressedMediaFrame0>();
    if (!mediaFrame)
        return true;

    const void* frameData = mediaFrame->data(0);
    if (!frameData)
        return true;

    int stride = videoFrame->lineSize(0);
    
    // CRITICAL: Clone the frame data immediately before returning
    cv::Mat frame(height, width, CV_8UC3, const_cast<void*>(frameData), stride);
    cv::Mat frameCopy = frame.clone();  // Make a copy we own
    
    // Queue for async processing
    {
        std::lock_guard<std::mutex> lock(m_queueMutex);
        
        // Drop old frames if queue is full (prevents lag buildup)
        if (m_frameQueue.size() >= 2)
        {
            NX_PRINT << "Frame queue full, dropping old frame";
            m_frameQueue.pop();
        }
        
        FrameData frameData;
        frameData.frame = frameCopy;
        frameData.timestampUs = videoFrame->timestampUs();
        frameData.frameIndex = m_frameIndex;
        frameData.hasMotion = true;
        
        m_frameQueue.push(frameData);
        m_queueCondition.notify_one();
    }
    
    // Return immediately - no lag!
    return true;
}

// Async processing thread
void DeviceAgent::processingThreadFunc()
{
    NX_PRINT << "Processing thread started";
    
    while (m_processingThreadRunning)
    {
        FrameData frameData;
        
        // Wait for frame
        {
            std::unique_lock<std::mutex> lock(m_queueMutex);
            m_queueCondition.wait(lock, [this] { 
                return !m_frameQueue.empty() || m_terminateThread; 
            });
            
            if (m_terminateThread)
                break;
            
            if (m_frameQueue.empty())
                continue;
            
            frameData = m_frameQueue.front();
            m_frameQueue.pop();
        }
        
        // Process frame
        processFrameAsync(frameData);
    }
    
    NX_PRINT << "Processing thread stopped";
}

void DeviceAgent::emitObjectDetectedEvent(
    const std::string& objectType,
    float confidence,
    const nx::sdk::Uuid& trackId,
    int64_t timestampUs)
{
    using namespace nx::sdk;
    using namespace nx::sdk::analytics;

    // Create analytics event
    auto event = makePtr<EventMetadata>();

   event->setTypeId("custom.object.detected");

    const std::string trackIdStr =
        nx::sdk::UuidHelper::toStdString(trackId);

    // const std::string confidenceStr =
    //     nx::sdk::format("%.3f", confidence);

    const std::string confidenceStr =
        std::to_string(confidence);

    const std::string timestampStr =
        std::to_string(timestampUs);

    // Caption → short & readable
    event->setCaption(
        "Object Detected | " + objectType
    );

    // Description → detailed, structured
    event->setDescription(
        "Type:" + objectType + ";" +
        "Confidence:" + confidenceStr + ";" +
        "TimestampUs:" + timestampStr + ";" 
    );

    event->setConfidence(confidence);
    event->setIsActive(true);
    event->setTrackId(trackId);

    // 🔑 Attributes → HTTP placeholders
    event->addAttribute(makePtr<Attribute>("objectType", objectType));
    event->addAttribute(makePtr<Attribute>("confidence", std::to_string(confidence)));
    event->addAttribute(makePtr<Attribute>(
        "trackId", UuidHelper::toStdString(trackId)));

    // 🔥 Wrap event in a METADATA PACKET
    auto eventPacket = makePtr<EventMetadataPacket>();
    eventPacket->setTimestampUs(timestampUs);
    eventPacket->addItem(event.get());

    // 🔥 THIS is what ConsumingDeviceAgent expects
    pushMetadataPacket(eventPacket.releasePtr());
}

void DeviceAgent::saveBestShotToDisk(
    const cv::Mat& image,
    const std::string& className,
    int64_t timestampUs,
    const nx::sdk::Uuid& trackId)
{
    try
    {
        if (image.empty())
        {
            NX_PRINT << "SKIP saving image: empty best shot";
            return;
        }

        // Camera ID (unique per DeviceAgent)
        const std::string& cameraId = m_cameraId;

        const std::string trackIdStr =
            nx::sdk::UuidHelper::toStdString(trackId);

        // Build paths
        std::filesystem::path cameraDir =
            std::filesystem::path(kDetectionsBasePath) /
            ("camera_" + cameraId);

        // std::filesystem::path trackDir =
        //     cameraDir / ("track_" + trackIdStr);

        // Create directories if not exist
        std::filesystem::create_directories(cameraDir);

        // sanitize class name (nx.base.Person → person)
        std::string cleanClass = className;
        std::transform(cleanClass.begin(), cleanClass.end(), cleanClass.begin(), ::tolower);
        std::replace(cleanClass.begin(), cleanClass.end(), '.', '_');

        // short track id (first 4 chars)
        std::string trackShort =
            nx::sdk::UuidHelper::toStdString(trackId).substr(0, 4);

        // filename
        std::string fileName =
            cleanClass + "_" +
            std::to_string(timestampUs) + "_" +
            trackShort + ".ppm";


        // Final image path
        std::filesystem::path imagePath = cameraDir / fileName;

        // Save image
        std::ofstream file(imagePath, std::ios::binary);
        if (!file.is_open())
        {
            NX_PRINT << "ERROR: Failed to open file "
                    << imagePath.string();
            return;
        }

        // Ensure BGR → RGB
        cv::Mat rgb;
        if (image.channels() == 3)
            cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
        else if (image.channels() == 1)
            cv::cvtColor(image, rgb, cv::COLOR_GRAY2RGB);
        else
        {
            NX_PRINT << "ERROR: Unsupported image format for disk save";
            return;
        }

        // Write PPM header
        file << "P6\n"
            << rgb.cols << " " << rgb.rows << "\n255\n";

        // Write raw pixel data
        file.write(reinterpret_cast<const char*>(rgb.data),
                rgb.total() * rgb.channels());

        file.close();

        NX_PRINT << "Saved best shot to: " << imagePath.string();


        NX_PRINT << "Saved best shot to: " << imagePath.string();
    }
    catch (const std::exception& e)
    {
        NX_PRINT << "ERROR saving best shot to disk: " << e.what();
    }
}


static std::string cocoToNxType(const std::string& coco)
{
    std::string out = "nx.custom.";
    for (unsigned char c : coco)
    {
        if (c == ' ')
            out += '_';
        else
            out += std::tolower(c);
    }
    return out;
}



void DeviceAgent::processFrameAsync(const FrameData& frameData)
{
    using namespace std::chrono;
    const auto startTime = high_resolution_clock::now();

    cv::Mat frame = frameData.frame;
    int width = frame.cols;
    int height = frame.rows;

    // Adaptive downscaling
    cv::Mat processFrame;
    float scale = 1.0f;
    if (width > 1280 || height > 720)
    {
        scale = std::min(1280.0f / width, 720.0f / height);
        cv::resize(frame, processFrame, cv::Size(), scale, scale, cv::INTER_LINEAR);
    }
    else
    {
        processFrame = frame;
    }

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes = runDetections(processFrame, classIds, confidences);

    // Scale boxes back
    if (scale != 1.0f)
    {
        for (auto& box : boxes)
        {
            box.x = static_cast<int>(box.x / scale);
            box.y = static_cast<int>(box.y / scale);
            box.width = static_cast<int>(box.width / scale);
            box.height = static_cast<int>(box.height / scale);
        }
    }

    // Exponential backoff logic
    if (m_isExponentialBackoffActive)
    {
        if (boxes.empty())
        {
            if (frameData.frameIndex % static_cast<int>(m_exponentialBackoffPeriod) == 0 &&
                m_exponentialBackoffPeriod < m_maxExponentialBackoffPeriod)
            {
                m_exponentialBackoffPeriod *= 1.1f;
            }
        }
        else
        {
            m_exponentialBackoffPeriod = static_cast<float>(m_objectDetectionPeriod);
            m_isExponentialBackoffActive = false;
        }
    }
    else
    {
        m_isExponentialBackoffActive = boxes.empty();
    }

    if (!boxes.empty())
        NX_PRINT << "Detected " << boxes.size() << " objects";

    // DEBUG: Draw bounding boxes on frame and save occasionally for verification
    static int debugFrameCount = 0;
    if (!boxes.empty() && debugFrameCount % 100 == 0)
    {
        try {
            cv::Mat debugFrame = frame.clone();
            for (size_t i = 0; i < boxes.size(); ++i)
            {
                const auto& box = boxes[i];
                std::string label = (classIds[i] < classNames.size()) ? classNames[classIds[i]] : "unknown";
                label += " " + std::to_string(static_cast<int>(confidences[i] * 100)) + "%";
                
                // Draw rectangle
                cv::rectangle(debugFrame, box, cv::Scalar(0, 255, 0), 2);
                
                // Draw label background
                int baseline = 0;
                cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
                cv::rectangle(debugFrame, 
                    cv::Point(box.x, box.y - textSize.height - 5),
                    cv::Point(box.x + textSize.width, box.y),
                    cv::Scalar(0, 255, 0), -1);
                
                // Draw label text
                cv::putText(debugFrame, label, 
                    cv::Point(box.x, box.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
            }
            
            // Try to save debug frame (will fail silently if no write permission)
            std::string debugPath = "/tmp/nx_detection_debug_" + std::to_string(frameData.frameIndex) + ".jpg";
            // Note: This might fail due to JPEG codec, but won't crash - just for debugging
            try {
                cv::imwrite(debugPath, debugFrame);
                NX_PRINT << "Saved debug frame to: " << debugPath;
            } catch (...) {
                // Silently ignore if save fails
            }
        } catch (...) {
            // Ignore debug drawing errors
        }
    }
    debugFrameCount++;

    // Create metadata packet
    auto packet = makePtr<ObjectMetadataPacket>();
    packet->setTimestampUs(frameData.timestampUs);

    // Process detections with tracking
    std::vector<Uuid> currentFrameTrackIds;
    
    {
        std::lock_guard<std::mutex> lock(m_trackingMutex);
        
        for (size_t i = 0; i < boxes.size(); ++i)
        {
            const auto& box = boxes[i];
            
            // IMPORTANT: Normalize coordinates for metadata (0.0 to 1.0)
            float x = static_cast<float>(box.x) / width;
            float y = static_cast<float>(box.y) / height;
            float w = static_cast<float>(box.width) / width;
            float h = static_cast<float>(box.height) / height;
            
            // Clamp to valid range [0, 1]
            x = std::max(0.0f, std::min(1.0f, x));
            y = std::max(0.0f, std::min(1.0f, y));
            w = std::max(0.0f, std::min(1.0f, w));
            h = std::max(0.0f, std::min(1.0f, h));
            
            // Make sure box doesn't exceed frame bounds
            if (x + w > 1.0f) w = 1.0f - x;
            if (y + h > 1.0f) h = 1.0f - y;

            // std::string cocoClassName = (classIds[i] < classNames.size()) ? classNames[classIds[i]] : "unknown";

            
            if (classIds[i] < 0 || classIds[i] >= static_cast<int>(classNames.size()))
            {
                NX_PRINT << "SKIP: invalid classId=" << classIds[i];
                continue;
            }

            std::string cocoClassName = normalizeClassName(classNames[classIds[i]]);

            NX_PRINT << "Normalized COCO [" << cocoClassName << "]";

            // // convert to lowercase
            // std::transform(
            //     cocoClassName.begin(),
            //     cocoClassName.end(),
            //     cocoClassName.begin(),
            //     [](unsigned char c) { return std::tolower(c); }
            // );

            std::string nxType = mapCocoClassToNxType(cocoClassName);
            NX_PRINT << "mapCocoClassToNxType: "<< nxType;

             // Log every detection
            NX_PRINT << "Raw detection: " << cocoClassName << " -> " << nxType 
                     << " @ confidence " << confidences[i];
    

            // FILTER 1: Skip unknown objects
            if (nxType == "nx.base.Unknown")
            {
                NX_PRINT << "  -> SKIP: Unknown type";
                continue;
            }

            // if (nxType == "nx.base.Unknown")
            // {
            //     nxType = "nx.custom." + cocoClassName;ss
            //     NX_PRINT << "  -> MAPPED: To custom type " << nxType;
            //     continue;
            // }

            
            // // FILTER 2: Only persons (OPTIONAL - comment out to detect all known types)
            // if (nxType != "nx.base.Person")
            // {
            //     NX_PRINT << "  -> SKIP: Not a Person";
            //     continue;
            // }

            
            // FILTER 3: Check region ONLY if explicitly enabled
            if (m_useDetectionRegion)
            {
                NX_PRINT << "  -> Checking detection region (enabled)...";
                if (!isBoxInRegion(box, width, height))
                {
                    NX_PRINT << "  -> SKIP: Outside detection region";
                    continue;
                }
                NX_PRINT << "  -> PASS: Inside detection region";
            }
            else
            {
                NX_PRINT << "  -> Detection region check SKIPPED (disabled)";
            }
            
            NX_PRINT << "  -> ACCEPT: Adding to metadata packet";

            // Track with PIXEL coordinates (not normalized)
            auto result = matchOrCreateTrack(box, nxType, frameData.timestampUs, frame, confidences[i]);
            Uuid trackId = result.trackId;
            currentFrameTrackIds.push_back(trackId);

            NX_PRINT << "Event Emitting Starting!";

            // if (result.isNew)
            // {
            //     NX_PRINT << "NEW TRACK → emitting event for " << nxType;

            //     emitObjectDetectedEvent(
            //         cocoClassName,
            //         confidences[i],
            //         trackId,
            //         frameData.timestampUs
            //     );
            // }

            auto now = frameData.timestampUs;

            auto it = std::find_if(
                m_trackedObjects.begin(),
                m_trackedObjects.end(),
                [&](const TrackedObject& t)
                {
                    return t.trackId == trackId;
                }
            );

            if (it != m_trackedObjects.end())
            {
                if (now - it->lastEventTimestampUs >= kEventIntervalUs)
                {
                    NX_PRINT << "Emitting periodic event for track " 
                            << nxType << " (" << trackId << ")";

                    emitObjectDetectedEvent(
                        cocoClassName,
                        confidences[i],
                        trackId,
                        now
                    );

                    it->lastEventTimestampUs = now;
                }
            }
            NX_PRINT << "Event Emitting Finished!";

            auto object = makePtr<ObjectMetadata>();
            object->setTypeId(nxType);
            object->setBoundingBox(Rect(x, y, w, h));  // Use normalized coordinates
            object->setTrackId(trackId);
            object->setConfidence(confidences[i]);
            
            // DEBUG: Log bounding box info
            if (frameData.frameIndex % 30 == 0)
            {
                NX_PRINT << "Object " << i << ": " << nxType 
                         << " at (" << x << ", " << y << ", " << w << ", " << h << ")"
                         << " pixel box: (" << box.x << ", " << box.y << ", " 
                         << box.width << ", " << box.height << ")";
            }

            packet->addItem(object.get());
        }

        // Clean up lost tracks periodically
        if (frameData.frameIndex - m_lastTrackCleanupFrame > 30)
        {
            cleanupLostTracks();
            m_lastTrackCleanupFrame = frameData.frameIndex;
        }

        pushMetadataPacket(packet.releasePtr());

        
        // Send best shot packets for newly detected tracks
        for (const Uuid& trackId : currentFrameTrackIds)
        {
            auto it = std::find_if(m_trackedObjects.begin(), m_trackedObjects.end(),
                [&trackId](const TrackedObject& t) { return t.trackId == trackId; });
            
           if (it != m_trackedObjects.end() &&
                !it->bestShotImage.empty() &&
                !it->bestShotSent)
            {
                // 🔥 Save to disk ONCE per track
                saveBestShotToDisk(
                    it->bestShotImage,
                    it->objectType,                    // class name
                    frameData.timestampUs,     // timestamp
                    trackId
                );

                // 🔥 Send to NX
                sendBestShotPacket(
                    trackId,
                    it->bestShotImage,
                    it->lastBox,
                    frameData.timestampUs,
                    width,
                    height
                );

                it->bestShotSent = true;  // ✅ sent & saved ONCE
            }
        }
    }


    const auto finishTime = high_resolution_clock::now();
    const auto duration = duration_cast<milliseconds>(finishTime - startTime);
    
    if (duration.count() > 100)
        NX_PRINT << "Frame processing duration: " << duration.count() << " ms";
}

void DeviceAgent::sendBestShotPacket(
    const Uuid& trackId,
    const cv::Mat& objectImage,
    const cv::Rect& boundingBox,
    int64_t timestampUs,
    int frameWidth,
    int frameHeight)
{
    if (objectImage.empty())
    {
        NX_PRINT << "WARNING: Best shot image is empty";
        return;
    }
    
    NX_PRINT << "Sending best shot: " << objectImage.cols << "x" << objectImage.rows 
             << " for box (" << boundingBox.x << ", " << boundingBox.y << ", "
             << boundingBox.width << ", " << boundingBox.height << ")";
    
    // Create best shot packet with CROPPED object image
    // Normalize bounding box coordinates (0.0 to 1.0)
    float x = static_cast<float>(boundingBox.x) / frameWidth;
    float y = static_cast<float>(boundingBox.y) / frameHeight;
    float w = static_cast<float>(boundingBox.width) / frameWidth;
    float h = static_cast<float>(boundingBox.height) / frameHeight;
    
    // Clamp to valid range
    x = std::max(0.0f, std::min(1.0f, x));
    y = std::max(0.0f, std::min(1.0f, y));
    w = std::max(0.0f, std::min(1.0f - x, w));
    h = std::max(0.0f, std::min(1.0f - y, h));
    
    auto bestShotPacket = makePtr<ObjectTrackBestShotPacket>(
        trackId,
        timestampUs,
        Rect(x, y, w, h)
    );
    
    // WORKAROUND: Convert to PPM format (simple, no codec needed!)
    std::vector<char> ppmData;
    
    try {
        // Ensure image is valid
        if (objectImage.cols <= 0 || objectImage.rows <= 0)
        {
            NX_PRINT << "ERROR: Invalid image dimensions";
            return;
        }
        
        // Ensure image is in BGR format
        cv::Mat bgrImage;
        if (objectImage.channels() == 3)
            bgrImage = objectImage;
        else if (objectImage.channels() == 1)
            cv::cvtColor(objectImage, bgrImage, cv::COLOR_GRAY2BGR);
        else if (objectImage.channels() == 4)
            cv::cvtColor(objectImage, bgrImage, cv::COLOR_BGRA2BGR);
        else
        {
            NX_PRINT << "ERROR: Unsupported image format with " << objectImage.channels() << " channels";
            return;
        }
        
        // Convert BGR to RGB (PPM uses RGB)
        cv::Mat rgbImage;
        cv::cvtColor(bgrImage, rgbImage, cv::COLOR_BGR2RGB);
        
        // Verify the conversion worked
        if (rgbImage.empty() || !rgbImage.isContinuous())
        {
            NX_PRINT << "ERROR: RGB conversion failed or image not continuous";
            // Make it continuous
            rgbImage = rgbImage.clone();
        }
        
        // Create PPM header (P6 = binary RGB)
        std::string header = "P6\n" + 
                           std::to_string(rgbImage.cols) + " " + 
                           std::to_string(rgbImage.rows) + "\n255\n";
        
        // Reserve space for efficiency
        size_t totalSize = header.size() + (rgbImage.total() * 3);
        ppmData.reserve(totalSize);
        
        // Add header
        ppmData.insert(ppmData.end(), header.begin(), header.end());
        
        // Add pixel data
        if (rgbImage.isContinuous())
        {
            const char* dataPtr = reinterpret_cast<const char*>(rgbImage.data);
            size_t dataSize = rgbImage.total() * rgbImage.channels();
            ppmData.insert(ppmData.end(), dataPtr, dataPtr + dataSize);
        }
        else
        {
            // Handle non-continuous image (row by row)
            for (int i = 0; i < rgbImage.rows; ++i)
            {
                const char* rowPtr = reinterpret_cast<const char*>(rgbImage.ptr(i));
                size_t rowSize = rgbImage.cols * rgbImage.channels();
                ppmData.insert(ppmData.end(), rowPtr, rowPtr + rowSize);
            }
        }
        
        if (!ppmData.empty())
        {
            bestShotPacket->setImageData(ppmData);
            pushMetadataPacket(bestShotPacket.releasePtr());
            NX_PRINT << "SUCCESS: Sent best shot (" 
                     << ppmData.size() << " bytes, " 
                     << rgbImage.cols << "x" << rgbImage.rows << ")";
        }
        else
        {
            NX_PRINT << "ERROR: PPM data is empty after creation";
        }
    }
    catch (const cv::Exception& e)
    {
        NX_PRINT << "ERROR: OpenCV exception in sendBestShotPacket: " << e.what();
    }
    catch (const std::exception& e)
    {
        NX_PRINT << "ERROR: Exception in sendBestShotPacket: " << e.what();
    }
}

void DeviceAgent::doSetNeededMetadataTypes(
    nx::sdk::Result<void>* /*outValue*/,
    const nx::sdk::analytics::IMetadataTypes* /*neededMetadataTypes*/)
{
}

void DeviceAgent::parseDetectionPolygon(const std::string& jsonValue)
{
    std::lock_guard<std::mutex> lock(m_polygonMutex);
    
    m_detectionPolygon.clear();
    // DON'T touch m_useDetectionRegion here - let settingsReceived() control it
    
    if (jsonValue.empty())
    {
        NX_PRINT << "Detection region polygon is empty";
        return;
    }
    
    try
    {
        std::string errors;
        nx::kit::Json polygonJson = nx::kit::Json::parse(jsonValue, errors);
        
        if (!errors.empty())
        {
            NX_PRINT << "ERROR: Failed to parse polygon JSON: " << errors;
            return;
        }
        
        // Extract points from JSON structure
        if (!polygonJson.is_object())
            return;
            
        auto figureObj = polygonJson["figure"];
        if (!figureObj.is_object())
            return;
            
        auto pointsArray = figureObj["points"];
        if (!pointsArray.is_array())
            return;
        
        for (const auto& point : pointsArray.array_items())
        {
            if (point.is_array() && point.array_items().size() >= 2)
            {
                float x = static_cast<float>(point[0].number_value());
                float y = static_cast<float>(point[1].number_value());
                m_detectionPolygon.push_back(cv::Point2f(x, y));
                NX_PRINT << "Polygon point: (" << x << ", " << y << ")";
            }
        }
        
        NX_PRINT << "Parsed polygon with " << m_detectionPolygon.size() << " points";
    }
    catch (const std::exception& e)
    {
        NX_PRINT << "EXCEPTION parsing detection polygon: " << e.what();
        m_detectionPolygon.clear();
    }
}

bool DeviceAgent::isPointInPolygon(const cv::Point2f& point, const std::vector<cv::Point2f>& polygon)
{
    if (polygon.size() < 3)
        return true;  // No polygon = detect everywhere
    
    // Ray casting algorithm
    bool inside = false;
    int j = polygon.size() - 1;
    
    for (size_t i = 0; i < polygon.size(); i++)
    {
        if (((polygon[i].y > point.y) != (polygon[j].y > point.y)) &&
            (point.x < (polygon[j].x - polygon[i].x) * (point.y - polygon[i].y) / 
                      (polygon[j].y - polygon[i].y) + polygon[i].x))
        {
            inside = !inside;
        }
        j = i;
    }
    
    return inside;
}

bool DeviceAgent::isBoxInRegion(const cv::Rect& box, int frameWidth, int frameHeight)
{
    std::lock_guard<std::mutex> lock(m_polygonMutex);
    
    if (!m_useDetectionRegion || m_detectionPolygon.empty())
        return true;  // No region defined = detect everywhere
    
    // Convert box to normalized coordinates
    float centerX = (box.x + box.width / 2.0f) / frameWidth;
    float centerY = (box.y + box.height / 2.0f) / frameHeight;
    
    cv::Point2f center(centerX, centerY);
    
    // Check if center point is inside polygon
    bool inside = isPointInPolygon(center, m_detectionPolygon);
    
    return inside;
}

nx::sdk::Result<const nx::sdk::ISettingsResponse*> DeviceAgent::settingsReceived()
{
    const std::lock_guard<std::mutex> lock(m_settingsMutex);

    NX_PRINT << "========== SETTINGS RECEIVED ==========";

    m_objectTypeIdsToGenerate.clear();

    std::string oldModelPath = m_modelPath;
    std::string oldClassesPath = m_classesPath;
    float oldConfidenceThreshold = m_confidenceThreshold;
    bool oldUseGpu = m_useGpu;

    const std::map<std::string, std::string>& settings = currentSettings();
    NX_PRINT << "Total settings count: " << settings.size();

    // FIRST PASS: Parse all settings and store them
    bool checkboxEnabled = false;
    std::string polygonValue;
    
    for (const auto& entry: settings)
    {
        const std::string& key = entry.first;
        const std::string& value = entry.second;
        
        NX_PRINT << "Setting: [" << key << "] = [" << value << "]";
        
        // Helper lambda to check if string starts with prefix
        auto startsWith = [](const std::string& str, const std::string& prefix) {
            return str.size() >= prefix.size() && 
                   str.compare(0, prefix.size(), prefix) == 0;
        };
        
        // Helper lambda to convert string to bool
        auto toBool = [](const std::string& str) {
            return str == "true" || str == "1";
        };
        
        if (key == kModelPathSetting)
            m_modelPath = value;
        else if (key == kClassesPathSetting)
            m_classesPath = value;
        else if (key == kConfidenceThresholdSetting)
            m_confidenceThreshold = std::stof(value) / 100.0f;
        else if (key == kNmsThresholdSetting)
            m_nmsThreshold = std::stof(value) / 100.0f;
        else if (key == kUseGpuSetting) 
        {
            m_useGpu = toBool(value);
            NX_PRINT << "Use GPU setting: " << (m_useGpu ? "ENABLED" : "DISABLED");
        }
        else if (key == kEnableDetectionRegionSetting)
        {
            checkboxEnabled = toBool(value);
            NX_PRINT << "Enable detection region checkbox: " << (checkboxEnabled ? "CHECKED" : "UNCHECKED");
        }
        else if (key == kDetectionRegionSetting)
        {
            polygonValue = value;
            NX_PRINT << "Received polygon data (length: " << value.length() << " chars)";
        }
        else if (key == "detectionFrequencyFps")
        {
            m_detectionFrequencyFps = std::stoi(value);
            m_objectDetectionPeriod = calcObjectDetectionPeriod();
            m_exponentialBackoffPeriod = static_cast<float>(m_objectDetectionPeriod);
        }
    }
    
    // SECOND PASS: Process polygon AFTER we know the checkbox state
    if (!polygonValue.empty())
    {
        parseDetectionPolygon(polygonValue);
    }
    
    // THIRD PASS: Apply the final state based on checkbox AND polygon
    {
        std::lock_guard<std::mutex> lock(m_polygonMutex);
        
        if (checkboxEnabled && m_detectionPolygon.size() >= 3)
        {
            m_useDetectionRegion = true;
            NX_PRINT << "Detection region ENABLED (checkbox=true, polygon has " << m_detectionPolygon.size() << " points)";
        }
        else
        {
            m_useDetectionRegion = false;
            if (!checkboxEnabled)
                NX_PRINT << "Detection region DISABLED (checkbox is unchecked)";
            else
                NX_PRINT << "Detection region DISABLED (polygon has insufficient points: " << m_detectionPolygon.size() << ")";
        }
    }
    
    NX_PRINT << "Parsed values:";
    NX_PRINT << "  Model Path: " << m_modelPath;
    NX_PRINT << "  Classes Path: " << m_classesPath;
    NX_PRINT << "  Confidence Threshold: " << m_confidenceThreshold;
    NX_PRINT << "  NMS Threshold: " << m_nmsThreshold;
    NX_PRINT << "  Detection Frequency FPS: " << m_detectionFrequencyFps;
    NX_PRINT << "  Use GPU: " << (m_useGpu ? "YES" : "NO"); 
    NX_PRINT << "  Detection Region Enabled: " << (m_useDetectionRegion ? "YES" : "NO");
    NX_PRINT << "  Polygon Points: " << m_detectionPolygon.size();

    if (m_modelPath != oldModelPath || m_classesPath != oldClassesPath || m_confidenceThreshold != oldConfidenceThreshold)
    {
        NX_PRINT << "Settings Changed, reloading model...";
        loadModel();
    }
    else
    {
        NX_PRINT << "Model paths unchanged, keeping existing model";
    }

    return nullptr;
}

Uuid DeviceAgent::trackIdByTrackIndex(int trackIndex)
{
    while (trackIndex >= m_trackIds.size())
        m_trackIds.push_back(nx::sdk::UuidHelper::randomUuid());

    return m_trackIds[trackIndex];
}

void DeviceAgent::loadModel()
{
    classNames.clear();

    std::string modelPath = "/opt/networkoptix/mediaserver/bin/plugins/stub_analytics_plugin/models/yolov5s.onnx";
    std::string classesPath = "/opt/networkoptix/mediaserver/bin/plugins/stub_analytics_plugin/models/classes.txt";

    NX_PRINT << "Attempting to load model from: " << m_modelPath;
    NX_PRINT << "Attempting to load classes from: " << m_classesPath;
    NX_PRINT << "Using " << (m_useGpu ? "GPU" : "CPU") << " for inference";

    // NX_PRINT << "Settings Recieved - modelPath: " << m_modelPath
    //          << ", classesPath: " << m_classesPath
    //          << ", confidenceThreshold: " << m_confidenceThreshold
    //          << ", nmsThreshold: " << m_nmsThreshold
    //          << ", detectionFrequencyFps: " << m_detectionFrequencyFps;

    try
    {
        std::ifstream modelFile(m_modelPath);
        if (!modelFile.good())
        {
            NX_PRINT << "ERROR: Model file not found at " << m_modelPath;
            return;
        }
        modelFile.close();

        std::ifstream classFile(m_classesPath);
        if (!classFile.good())
        {
            NX_PRINT << "ERROR: Classes file not found at " << m_classesPath;
            return;
        }

        std::string line;
        while (getline(classFile, line))
        {
            if (!line.empty())
                classNames.push_back(line);
        }

        NX_PRINT << "Loaded classes:";
        for (size_t i = 0; i < classNames.size(); ++i)
            NX_PRINT << i << " -> " << classNames[i];

        classFile.close();
        
        NX_PRINT << "Loaded " << classNames.size() << " class names";

        yoloNet = cv::dnn::readNetFromONNX(modelPath);


        // yoloNet.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        // yoloNet.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        
        if (m_useGpu)
        {
            try {
                yoloNet.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                yoloNet.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
                NX_PRINT << "GPU (CUDA) backend enabled";
            } catch (const cv::Exception& e) {
                NX_PRINT << "WARNING: Failed to set CUDA backend: " << e.what();
                NX_PRINT << "Falling back to CPU";
                yoloNet.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                yoloNet.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
                m_useGpu = false;  // Update flag to reflect actual state
            }
        }
        else
        {
            yoloNet.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            yoloNet.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            NX_PRINT << "CPU backend enabled";
        }

        if (yoloNet.empty())
        {
            NX_PRINT << "ERROR: YOLO network is empty after loading!";
            return;
        }

        NX_PRINT << "YOLO model loaded successfully!";
    }
    catch (const std::exception& e)
    {
        NX_PRINT << "EXCEPTION while loading YOLO model: " << e.what();
    }
}

std::vector<cv::Rect> DeviceAgent::runDetections(
    const cv::Mat& frame, std::vector<int>& classIds, std::vector<float>& confidences)
{
    // float confThreshold = 0.75f;
    // float nmsThreshold  = 0.45f;



    float confThreshold = m_confidenceThreshold;
    float nmsThreshold  = m_nmsThreshold;

    std::vector<cv::Rect> boxes;
    classIds.clear();
    confidences.clear();

    try
    {
        // Prepare blob
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0,
            cv::Size(640, 640), cv::Scalar(), true, false);

        yoloNet.setInput(blob);

        std::vector<cv::Mat> outputs;
        yoloNet.forward(outputs, yoloNet.getUnconnectedOutLayersNames());

        if (outputs.empty())
            return boxes;
// tub_analytics_plugin/models/classes.txt
        cv::Mat output = outputs[0];
        if (output.dims != 3)
            return boxes;

        int numDetections = output.size[1];
        int numParams = output.size[2];

        cv::Mat detections = output.reshape(1, numDetections);

        std::vector<cv::Rect> tempBoxes;
        std::vector<int> tempClassIds;
        std::vector<float> tempConfidences;

        int highConfCount = 0;

        for (int i = 0; i < detections.rows; i++)
        {
            float confidence = detections.at<float>(i, 4);

            if (confidence > confThreshold)
                highConfCount++;

            if (confidence > confThreshold)
            {
                cv::Mat scores = detections.row(i).colRange(5, numParams);
                cv::Point classIdPoint;
                double maxClassScore;
                cv::minMaxLoc(scores, 0, &maxClassScore, 0, &classIdPoint);

                float combinedScore = confidence * maxClassScore;

                if (combinedScore > confThreshold)
                {
                    // YOLOv5 raw coords in 640x640
                    float cx = detections.at<float>(i, 0);
                    float cy = detections.at<float>(i, 1);
                    float w = detections.at<float>(i, 2);
                    float h = detections.at<float>(i, 3);

                    // scale to frame size
                    float scaleX = float(frame.cols) / 640.0f;
                    float scaleY = float(frame.rows) / 640.0f;

                    float scaledCx = cx * scaleX;
                    float scaledCy = cy * scaleY;
                    float scaledW = w * scaleX;
                    float scaledH = h * scaleY;

                    int left = int(scaledCx - scaledW / 2);
                    int top = int(scaledCy - scaledH / 2);
                    int width = int(scaledW);
                    int height = int(scaledH);

                    // clamp
                    left = std::max(0, std::min(left, frame.cols - 1));
                    top  = std::max(0, std::min(top, frame.rows - 1));
                    width = std::min(width, frame.cols - left);
                    height = std::min(height, frame.rows - top);

                    if (width > 0 && height > 0)
                    {
                        tempBoxes.emplace_back(left, top, width, height);
                        tempClassIds.push_back(classIdPoint.x);
                        tempConfidences.push_back(combinedScore);
                    }
                }
            }
        }

        NX_PRINT << "Candidates with conf > " << confThreshold << ": " << highConfCount;

        if (tempBoxes.empty())
            return boxes;

        NX_PRINT << "Before NMS: " << tempBoxes.size() << " detections";

        std::vector<int> indices;
        cv::dnn::NMSBoxes(tempBoxes, tempConfidences,
                          confThreshold, nmsThreshold, indices);

        NX_PRINT << "After NMS: " << indices.size() << " detections";

        for (int idx : indices)
        {
            boxes.push_back(tempBoxes[idx]);
            classIds.push_back(tempClassIds[idx]);
            confidences.push_back(tempConfidences[idx]);
        }
    }
    catch (const std::exception& e)
    {
        NX_PRINT << "EXCEPTION in runDetections: " << e.what();
    }

    return boxes;
}


} // namespace object_detection
} // namespace stub
} // namespace analytics
} // namespace vms_server_plugins
} // namespace nx