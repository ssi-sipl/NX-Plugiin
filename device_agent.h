// device_agent.h - Updated header with async processing

#pragma once

#include <nx/sdk/analytics/helpers/consuming_device_agent.h>
#include <nx/sdk/helpers/uuid_helper.h>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <set>
#include <vector>
#include <string>
#include <map>
#include <nx/kit/json.h>

namespace nx {
namespace vms_server_plugins {
namespace analytics {
namespace stub {
namespace object_detection {

struct TrackedObject;
struct FrameData;

struct TrackMatchResult
{
    nx::sdk::Uuid trackId;
    bool isNew;
};


class DeviceAgent: public nx::sdk::analytics::ConsumingDeviceAgent
{
public:
    DeviceAgent(const nx::sdk::IDeviceInfo* deviceInfo);
    virtual ~DeviceAgent() override;

protected:
    virtual std::string manifestString() const override;
    
    virtual bool pushUncompressedVideoFrame(
        const nx::sdk::analytics::IUncompressedVideoFrame* videoFrame) override;

    virtual void doSetNeededMetadataTypes(
        nx::sdk::Result<void>* outValue,
        const nx::sdk::analytics::IMetadataTypes* neededMetadataTypes) override;

    virtual nx::sdk::Result<const nx::sdk::ISettingsResponse*> settingsReceived() override;

private:
    // Frame counter
    int m_frameIndex = 0;

    std::string m_cameraId;


    // Plugin Settings
    std::string m_modelPath;
    std::string m_classesPath;
    float m_confidenceThreshold = 0.50f;
    float m_nmsThreshold = 0.45f;
    bool m_useGpu = false;
    
    // Detection region polygon
    std::vector<cv::Point2f> m_detectionPolygon;  // Stores polygon points (normalized 0-1)
    bool m_useDetectionRegion;  // Flag to enable/disable region filtering
    std::mutex m_polygonMutex;  // Protect polygon access
    
    // Settings
    std::mutex m_settingsMutex;
    std::set<std::string> m_objectTypeIdsToGenerate;
    bool m_sendAttributes = false;
    int m_timestampShiftMs = 0;
    std::vector<nx::sdk::Uuid> m_trackIds;
    
    // OpenCV YOLO
    cv::dnn::Net yoloNet;
    std::vector<std::string> classNames;
    
    // Tracking
    std::mutex m_trackingMutex;
    std::vector<TrackedObject> m_trackedObjects;
    
    // FPS and detection management
    float m_fps;
    int m_detectionFrequencyFps;
    int m_objectDetectionPeriod;
    int64_t m_fpsChangeTimestampUs;
    
    // Exponential backoff
    float m_exponentialBackoffPeriod;
    float m_maxExponentialBackoffPeriod;
    bool m_isExponentialBackoffActive;
    
    // Tracking cleanup
    int64_t m_lastTrackCleanupFrame;
    
    // ASYNC PROCESSING - Critical for eliminating lag!
    std::thread m_processingThread;
    std::queue<FrameData> m_frameQueue;
    std::mutex m_queueMutex;
    std::condition_variable m_queueCondition;
    bool m_processingThreadRunning;
    bool m_terminateThread;

    // Detection region methods
    void parseDetectionPolygon(const std::string& jsonValue);
    bool isPointInPolygon(const cv::Point2f& point, const std::vector<cv::Point2f>& polygon);
    bool isBoxInRegion(const cv::Rect& box, int frameWidth, int frameHeight);
    
    // Thread function
    void processingThreadFunc();
    void processFrameAsync(const FrameData& frameData);
    
    // Model loading and detection
    void loadModel();
    std::vector<cv::Rect> runDetections(
        const cv::Mat& frame,
        std::vector<int>& classIds,
        std::vector<float>& confidences);
    
    // Helper methods
    std::string mapCocoClassToNxType(const std::string& cocoClass);
    nx::sdk::Uuid trackIdByTrackIndex(int trackIndex);
    
    // Tracking methods
    float calculateIoU(const cv::Rect& box1, const cv::Rect& box2);
    float calculateImageQuality(const cv::Mat& image, const cv::Rect& box);
    TrackMatchResult matchOrCreateTrack(
        const cv::Rect& box,
        const std::string& objectType,
        int64_t timestamp,
        const cv::Mat& frame,
        float confidence);
    void cleanupLostTracks();
    
    // Best shot handling
    void sendBestShotPacket(
        const nx::sdk::Uuid& trackId,
        const cv::Mat& objectImage,
        const cv::Rect& boundingBox,
        int64_t timestampUs,
        int frameWidth,
        int frameHeight);
    
    // Optimization methods
    bool needToDetectObjects(const nx::sdk::analytics::IUncompressedVideoFrame* videoFrame);
    void updateFpsAndDetectionPeriod(int64_t timestampUs);
    int calcObjectDetectionPeriod() const;

    void emitObjectDetectedEvent(
        const std::string& objectType,
        float confidence,
        const nx::sdk::Uuid& trackId,
        int64_t timestampUs);
    
    void saveBestShotToDisk(
        const cv::Mat& image,
        const std::string& className,
        int64_t timestampUs,
        const nx::sdk::Uuid& trackId);


    
public:
    static const std::string kTimeShiftSetting;
    static const std::string kSendAttributesSetting;
    static const std::string kObjectTypeGenerationSettingPrefix;
    static const std::string kModelPathSetting;
    static const std::string kClassesPathSetting;
    static const std::string kConfidenceThresholdSetting;
    static const std::string kNmsThresholdSetting;
    static const std::string kDetectionRegionSetting;
    static const std::string kEnableDetectionRegionSetting;
    static const std::string kUseGpuSetting;
};

} // namespace object_detection
} // namespace stub
} // namespace analytics
} // namespace vms_server_plugins
} // namespace nx