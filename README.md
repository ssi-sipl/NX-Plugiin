# NX-Plugiin

StubAnalyticsPlugin -

1. Install the MetaDataSDK - https://nxvms.com/download/releases/sdk
2. Go to this location - /nxwitness-metadata_sdk-6.0.6.41837-universal/metadata_sdk/samples/stub_analytics_plugin
3. Install these dependencies -
   - sudo apt update
   - sudo apt install build-essential cmake g++
4. Build for the first time -
   - python3 -m venv venv
   - source venv/bin/activate
   - pip install conan==1.62.0
   - mkdir build
   - cd build
   - cmake .. -DCMAKE_BUILD_TYPE=Release
   - make
5. Then in the /build directory you will find the .so file
6. Copy the .so file to the plugins directory
   - sudo cp libstub_analytics_plugin.so /opt/networkoptix/mediaserver/bin/plugins/
7. Then restart the nx witness
   - sudo systemctl restart networkoptix-mediaserver.service
8. To see the logs - sudo systemctl status networkoptix-mediaserver.service - sudo journalctl -u networkoptix-mediaserver.service -f

Defaults - /opt/networkoptix/mediaserver/bin/plugins/stub_analytics_plugin/models/yolov5s.onnx - /opt/networkoptix/mediaserver/bin/plugins/stub_analytics_plugin/models/classes.txt

9. Creating the Rule -
   When - Analytics Event
   Event Type - Object Detected
   API Endpoint - http://localhost:5000/api/alerts/from-nx
   Content Type - application/json
   Body - {
   "sensorId":"ipcam1",
   "data":"{event.description}"
   }

   All Possible Fields - {
   "caption":{event.caption}
   "description":{event.description}
   "cameraId":{event.cameraId}
   "cameraName":{event.cameraName}
   "eventName":{event.eventName}
   }
