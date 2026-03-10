// Copyright 2018-present Network Optix, Inc. Licensed under MPL 2.0: www.mozilla.org/MPL/2.0/

#include "engine.h"
#include "device_agent.h"
#include "device_agent_manifest.h"
#include "stub_analytics_plugin_object_detection_ini.h"
#include <nx/kit/json.h>
#include <cctype>

namespace nx {
namespace vms_server_plugins {
namespace analytics {
namespace stub {
namespace object_detection {

using namespace nx::sdk;
using namespace nx::sdk::analytics;

static const std::set<std::string> kObjectTypeIdsGeneratedByDefault = {
    "nx.base.Bike",
    "nx.base.Bus",
    "nx.base.LicensePlate",
    "nx.base.Face",
    "nx.base.Person"
};

// static const std::string kModelPathSetting;
// static const std::string kClassesPathSetting;
// static const std::string kConfidenceThresholdSetting;
// static const std::string kNmsThresholdSetting;


Engine::Engine(): nx::sdk::analytics::Engine(ini().enableOutput)
{
}

Engine::~Engine()
{
}

void Engine::doObtainDeviceAgent(Result<IDeviceAgent*>* outResult, const IDeviceInfo* deviceInfo)
{
    *outResult = new DeviceAgent(deviceInfo);
}

static std::string makeCaption(const std::string& objectTypeId)
{
    std::string name = objectTypeId;

    // Remove prefix nx.base.
    const std::string prefix = "nx.base.";
    if (name.rfind(prefix, 0) == 0)
        name = name.substr(prefix.length());

    // Insert spaces before capital letters
    std::string caption;
    for (size_t i = 0; i < name.size(); ++i)
    {
        if (i > 0 && std::isupper(name[i]))
            caption += " ";

        caption += name[i];
    }

    return caption;
}

std::string Engine::manifestString() const
{
    using namespace nx::kit;

    std::string errors;
    Json deviceAgentManifest = Json::parse(kDeviceAgentManifest, errors);


    Json::array generationSettings;

    Json::object timeShiftSetting = {
        {"type", "SpinBox"},
        {"name", DeviceAgent::kTimeShiftSetting},
        {"caption", "Timestamp shift"},
        {"description", "Metadata timestamp shift in milliseconds"},
        {"defaultValue", 0}
    };
    // generationSettings.push_back(std::move(timeShiftSetting));

    Json::object attributesSetting = {
        {"type", "CheckBox"},
        {"name", DeviceAgent::kSendAttributesSetting},
        {"caption", "Send object attributes"},
        {"defaultValue", true}
    };

    // generationSettings.push_back(std::move(attributesSetting));

    // Change
    Json::object modelPathSetting = {
        {"type", "TextField"},
        {"name", DeviceAgent::kModelPathSetting},
        {"caption", "Model file path"},
        {"defaultValue", "/opt/networkoptix/mediaserver/bin/plugins/stub_analytics_plugin/models/openvino/yolov5su.xml"}
    };
    generationSettings.push_back(std::move(modelPathSetting));
    Json::object classesPathSetting = {
        {"type", "TextField"},
        {"name", DeviceAgent::kClassesPathSetting},
        {"caption", "Classes file path"},
        {"defaultValue", "/opt/networkoptix/mediaserver/bin/plugins/stub_analytics_plugin/models/classes.txt"}
    };
    generationSettings.push_back(std::move(classesPathSetting));
    Json::object confidenceThresholdSetting = {
        {"type", "SpinBox"},
        {"name", DeviceAgent::kConfidenceThresholdSetting},
        {"caption", "Confidence threshold"},
        {"minValue", 1},
        {"maxValue", 100},
        {"defaultValue", 75}
        // {"singleStep", 0.01}
    };
    generationSettings.push_back(std::move(confidenceThresholdSetting));
    Json::object nmsThresholdSetting = {
        {"type", "SpinBox"},
        {"name", DeviceAgent::kNmsThresholdSetting},
        {"caption", "NMS threshold"},
        {"minValue", 1},
        {"maxValue", 100},
        {"defaultValue", 45}
        // {"singleStep", 0.01}
    };
    generationSettings.push_back(std::move(nmsThresholdSetting));

    generationSettings.push_back(Json::object{ {"type", "Separator"} });

    Json::object useGpuSetting = {
        {"type", "CheckBox"},
        {"name", DeviceAgent::kUseGpuSetting},
        {"caption", "Use GPU acceleration"},
        {"description", "Use OpenVINO GPU device if available (Intel GPU recommended)"},
        {"defaultValue", false}
    };
    generationSettings.push_back(std::move(useGpuSetting));

    generationSettings.push_back(Json::object{ {"type", "Separator"} });

    // ADD THIS CHECKBOX SETTING
    Json::object enableRegionSetting = {
        {"type", "CheckBox"},
        {"name", DeviceAgent::kEnableDetectionRegionSetting},
        {"caption", "Enable Detection Region"},
        {"description", "Check this box to enable polygon-based region filtering"},
        {"defaultValue", false}
    };
    generationSettings.push_back(std::move(enableRegionSetting));
    
    // ADD POLYGON SETTING HERE
    Json::object detectionRegionSetting = {
        {"type", "PolygonFigure"},
        {"name", DeviceAgent::kDetectionRegionSetting},  // We'll define this constant
        {"caption", "Detection Region"},
        {"description", "Draw a polygon to define the detection area. Only objects inside this region will be detected."},
        {"minPoints", 3},
        {"maxPoints", 10}
    };
    generationSettings.push_back(std::move(detectionRegionSetting));
    

    // End Change


    generationSettings.push_back(Json::object{ {"type", "Separator"} });

    // generationSettings.push_back(Json::object{
    //     {"type", "Label"},
    //     {"text", "Object Types To Detect"}
    // });

    // for (const auto& supportedType : deviceAgentManifest["supportedTypes"].array_items())
    // {
    //     Json::object supportedTypeObject = supportedType.object_items();
    //     const std::string& objectTypeId = supportedTypeObject["objectTypeId"].string_value();

    //     Json::object generationSetting = {
    //         {"type", "CheckBox"},
    //         {"name", DeviceAgent::kObjectTypeGenerationSettingPrefix + objectTypeId},
    //         {"caption", makeCaption(objectTypeId)},
    //         {
    //             "defaultValue",
    //             kObjectTypeIdsGeneratedByDefault.find(objectTypeId)
    //                 != kObjectTypeIdsGeneratedByDefault.cend()
    //         }
    //     };

    //     generationSettings.push_back(std::move(generationSetting));
    // }

    Json::object settingsModel = {
        {"type", "Settings"},
        {"items", generationSettings}
    };

    Json::object engineManifest = {
        {"capabilities", "needUncompressedVideoFrames_bgr"},
        {"streamTypeFilter", "uncompressedVideo"},
        {"deviceAgentSettingsModel", settingsModel}
    };

    return Json(engineManifest).dump();
}

} // namespace object_detection
} // namespace stub
} // namespace analytics
} // namespace vms_server_plugins
} // namespace nx
