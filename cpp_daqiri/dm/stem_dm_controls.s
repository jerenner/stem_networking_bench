// Phase-one modeless STEM DAQ control palette for GMS 3.
//
// The palette never opens a network socket. Button callbacks copy typed values
// to persistent tags and increment Request:Sequence. stem_dm_viewer.py consumes
// the request between image polls and writes current state back to the tags.

string STEMRoot() { return "STEM DAQ"; }

string STEMReadString(string path, string fallback)
{
    string value;
    if (GetPersistentTagGroup().TagGroupGetTagAsString(path, value))
        return value;
    return fallback;
}

number STEMReadNumber(string path, number fallback)
{
    number value;
    if (GetPersistentTagGroup().TagGroupGetTagAsNumber(path, value))
        return value;
    return fallback;
}

number STEMReadBoolean(string path, number fallback)
{
    number value;
    if (GetPersistentTagGroup().TagGroupGetTagAsBoolean(path, value))
        return value;
    return fallback;
}

number STEMStageIndex(string stage)
{
    if (stage == "raw") return 0;
    if (stage == "dark_subtracted") return 1;
    if (stage == "dark_blr") return 2;
    if (stage == "corrected") return 3;
    if (stage == "thresholded") return 4;
    return 3;
}

string STEMStageName(number index)
{
    if (index == 0) return "raw";
    if (index == 1) return "dark_subtracted";
    if (index == 2) return "dark_blr";
    if (index == 3) return "corrected";
    if (index == 4) return "thresholded";
    return "corrected";
}

taggroup STEMStageChoice(string identifier, string initialStage)
{
    taggroup entries;
    taggroup choice = DLGCreateChoice(entries, STEMStageIndex(initialStage)).DLGIdentifier(identifier);
    choice.DLGAddChoiceItemEntry("raw");
    choice.DLGAddChoiceItemEntry("dark_subtracted");
    choice.DLGAddChoiceItemEntry("dark_blr");
    choice.DLGAddChoiceItemEntry("corrected");
    choice.DLGAddChoiceItemEntry("thresholded");
    return choice;
}

taggroup STEMLabeledField(string label, taggroup field)
{
    return DLGGroupItems(DLGCreateLabel(label, 24), field);
}

class STEMDAQControlPalette : UIFrame
{
    void RequestViewerStop(object self)
    {
        GetPersistentTagGroup().TagGroupSetTagAsBoolean(STEMRoot() + ":Viewer:StopRequested", 1);
    }

    ~STEMDAQControlPalette(object self)
    {
        self.RequestViewerStop();
    }

    void QueueCommand(object self, string command)
    {
        taggroup tags = GetPersistentTagGroup();
        number sequence = 0;
        tags.TagGroupGetTagAsLong(STEMRoot() + ":Control:Request:Sequence", sequence);
        tags.TagGroupSetTagAsString(STEMRoot() + ":Control:Request:Command", command);
        // Sequence is committed last so Python cannot observe partial fields.
        tags.TagGroupSetTagAsLong(STEMRoot() + ":Control:Request:Sequence", sequence + 1);
        self.DLGValue("status-message", "Queued: " + command);
    }

    void RefreshStatus(object self)
    {
        string base = STEMRoot() + ":State:";
        number online = STEMReadBoolean(base + "EngineOnline", 0);
        string engineStatus = "offline";
        if (online) engineStatus = "online";
        self.DLGValue("status-engine", engineStatus);
        self.DLGValue("status-control", STEMReadString(base + "Control", "waiting"));
        self.DLGValue("status-acquisition", STEMReadString(base + "Acquisition", "unknown"));
        self.DLGValue("status-visualization", STEMReadString(base + "Visualization", "unknown"));
        self.DLGValue("status-burst", STEMReadString(base + "Burst", "unknown"));
        self.DLGValue("status-instrument", STEMReadString(base + "Instrument", "unknown"));
        self.DLGValue("status-message", STEMReadString(base + "Message", ""));
    }

    void ReloadSettings(object self)
    {
        string visual = STEMRoot() + ":Control:Visualization:";
        string burst = STEMRoot() + ":Control:Burst:";
        string camera = STEMRoot() + ":Control:Camera:";
        string scan = STEMRoot() + ":Control:Scan:";
        string cameraService = "offline";
        string detectorSync = "no";
        string detectorAligned = "no";
        if (STEMReadBoolean(camera + "ServiceOnline", 0)) cameraService = "online";
        if (STEMReadBoolean(camera + "DetectorSynchronized", 0)) detectorSync = "yes";
        if (STEMReadBoolean(camera + "DetectorAligned", 0)) detectorAligned = "yes";
        self.DLGValue("viz-publishing", STEMReadBoolean(visual + "Publishing", 1));
        self.DLGValue("viz-stage", STEMStageIndex(STEMReadString(visual + "ProcessingStage", "corrected")));
        self.DLGValue("viz-rate", STEMReadNumber(visual + "RefreshHz", 1.0));
        self.DLGValue("viz-representative", STEMReadNumber(visual + "RepresentativeFrame", 64));
        self.DLGValue("viz-include-representative", STEMReadBoolean(visual + "IncludeRepresentative", 1));
        self.DLGValue("viz-include-sum", STEMReadBoolean(visual + "IncludeSum", 1));
        self.DLGValue("viz-zlp", STEMReadNumber(visual + "ZLPThreshold", 0));
        self.DLGValue("viz-core", STEMReadNumber(visual + "CoreLossThreshold", 0));

        self.DLGValue("burst-stage", STEMStageIndex(STEMReadString(burst + "ProcessingStage", "corrected")));
        self.DLGValue("burst-file", STEMReadString(burst + "FilepathTemplate", "/data/stem_burst_rx{receiver}_{capture}_{stage}.h5"));
        self.DLGValue("burst-dataset", STEMReadString(burst + "DatasetName", "/frames"));
        self.DLGValue("burst-buckets", STEMReadNumber(burst + "BucketsPerCapture", 1));
        self.DLGValue("burst-count", STEMReadNumber(burst + "CaptureCount", 1));
        self.DLGValue("burst-rearm", STEMReadBoolean(burst + "RearmAfterWrite", 1));
        self.DLGValue("burst-strict", STEMReadBoolean(burst + "StrictComplete", 0));
        self.DLGValue("burst-zlp", STEMReadNumber(burst + "ZLPThreshold", 0));
        self.DLGValue("burst-core", STEMReadNumber(burst + "CoreLossThreshold", 0));

        self.DLGValue("camera-service", cameraService);
        self.DLGValue("camera-mode", STEMReadString(camera + "Mode", "unknown"));
        self.DLGValue("camera-power", STEMReadString(camera + "PowerState", "unknown"));
        self.DLGValue("camera-insertion", STEMReadString(camera + "InsertionState", "unknown"));
        self.DLGValue("camera-temperature", STEMReadNumber(camera + "TemperatureC", 0));
        self.DLGValue("camera-target", STEMReadNumber(camera + "TargetTemperatureC", 0));
        self.DLGValue("detector-links", STEMReadString(camera + "DetectorLinks", "unknown"));
        self.DLGValue("detector-sync", detectorSync);
        self.DLGValue("detector-aligned", detectorAligned);
        self.DLGValue("operation-name", STEMReadString(camera + "OperationName", ""));
        self.DLGValue("operation-state", STEMReadString(camera + "OperationState", "idle"));
        self.DLGValue("operation-step", STEMReadString(camera + "OperationStep", ""));
        self.DLGValue("operation-completed", STEMReadNumber(camera + "OperationCompletedSteps", 0));
        self.DLGValue("operation-total", STEMReadNumber(camera + "OperationTotalSteps", 0));
        self.DLGValue("operation-error", STEMReadString(camera + "OperationError", ""));

        self.DLGValue("scan-state", STEMReadString(scan + "State", "idle"));
        self.DLGValue("scan-number", STEMReadNumber(scan + "ScanNumber", 0));
        self.DLGValue("scan-expected", STEMReadNumber(scan + "ExpectedFrames", 0));
        self.DLGValue("scan-received", STEMReadNumber(scan + "ReceivedFrames", 0));
        self.DLGValue("scan-pause", STEMReadNumber(scan + "PauseCount", 200));
        self.DLGValue("scan-read", STEMReadNumber(scan + "ReadCount", 1));
        self.DLGValue("scan-x", STEMReadNumber(scan + "PositionsX", 1));
        self.DLGValue("scan-rows", STEMReadNumber(scan + "Rows", 1));
        self.DLGValue("scan-flyback", STEMReadNumber(scan + "Flyback", 100));
        self.DLGValue("scan-flush", STEMReadBoolean(scan + "FlushMemory", 1));
        self.RefreshStatus();
    }

    void OnStart(object self) { self.QueueCommand("start_acquisition"); }
    void OnStop(object self) { self.QueueCommand("stop_acquisition"); }
    void OnRefresh(object self) { self.ReloadSettings(); }
    void OnStopViewer(object self) { self.RequestViewerStop(); self.Close(); }

    void StoreVisualization(object self)
    {
        taggroup tags = GetPersistentTagGroup();
        string root = STEMRoot() + ":Control:Visualization:";
        tags.TagGroupSetTagAsBoolean(root + "Publishing", self.LookupElement("viz-publishing").DLGGetValue());
        tags.TagGroupSetTagAsString(root + "ProcessingStage", STEMStageName(self.LookupElement("viz-stage").DLGGetValue()));
        tags.TagGroupSetTagAsNumber(root + "RefreshHz", self.LookupElement("viz-rate").DLGGetValue());
        tags.TagGroupSetTagAsLong(root + "RepresentativeFrame", self.LookupElement("viz-representative").DLGGetValue());
        tags.TagGroupSetTagAsBoolean(root + "IncludeRepresentative", self.LookupElement("viz-include-representative").DLGGetValue());
        tags.TagGroupSetTagAsBoolean(root + "IncludeSum", self.LookupElement("viz-include-sum").DLGGetValue());
        tags.TagGroupSetTagAsNumber(root + "ZLPThreshold", self.LookupElement("viz-zlp").DLGGetValue());
        tags.TagGroupSetTagAsNumber(root + "CoreLossThreshold", self.LookupElement("viz-core").DLGGetValue());
    }

    void OnApplyVisualization(object self)
    {
        self.StoreVisualization();
        self.QueueCommand("apply_visualization");
    }

    void StoreBurst(object self)
    {
        taggroup tags = GetPersistentTagGroup();
        string root = STEMRoot() + ":Control:Burst:";
        tags.TagGroupSetTagAsString(root + "ProcessingStage", STEMStageName(self.LookupElement("burst-stage").DLGGetValue()));
        tags.TagGroupSetTagAsString(root + "FilepathTemplate", self.LookupElement("burst-file").DLGGetStringValue());
        tags.TagGroupSetTagAsString(root + "DatasetName", self.LookupElement("burst-dataset").DLGGetStringValue());
        tags.TagGroupSetTagAsLong(root + "BucketsPerCapture", self.LookupElement("burst-buckets").DLGGetValue());
        tags.TagGroupSetTagAsLong(root + "CaptureCount", self.LookupElement("burst-count").DLGGetValue());
        tags.TagGroupSetTagAsBoolean(root + "RearmAfterWrite", self.LookupElement("burst-rearm").DLGGetValue());
        tags.TagGroupSetTagAsBoolean(root + "StrictComplete", self.LookupElement("burst-strict").DLGGetValue());
        tags.TagGroupSetTagAsNumber(root + "ZLPThreshold", self.LookupElement("burst-zlp").DLGGetValue());
        tags.TagGroupSetTagAsNumber(root + "CoreLossThreshold", self.LookupElement("burst-core").DLGGetValue());
    }

    void OnConfigureBurst(object self)
    {
        self.StoreBurst();
        self.QueueCommand("configure_burst");
    }

    void OnArmBurst(object self)
    {
        self.StoreBurst();
        self.QueueCommand("arm_burst");
    }

    void OnDisarmBurst(object self) { self.QueueCommand("disarm_burst"); }
    void OnAbortBurst(object self) { self.QueueCommand("abort_burst"); }
    void OnInstrumentRefresh(object self) { self.QueueCommand("refresh_instrument"); }
    void OnCameraTemperature(object self) { self.QueueCommand("camera_read_temperature"); }
    void OnCameraBiases(object self) { self.QueueCommand("camera_read_biases"); }
    void OnCameraPowerUp(object self) { self.QueueCommand("camera_power_up"); }
    void OnCameraPowerDown(object self) { self.QueueCommand("camera_power_down"); }
    void OnCameraInsert(object self) { self.QueueCommand("camera_insert"); }
    void OnCameraRetract(object self) { self.QueueCommand("camera_retract"); }
    void OnDetectorLinks(object self) { self.QueueCommand("detector_read_links"); }
    void OnDetectorResync(object self) { self.QueueCommand("detector_resync"); }
    void OnDetectorAutoAlign(object self) { self.QueueCommand("detector_auto_align"); }

    void StoreScan(object self)
    {
        taggroup tags = GetPersistentTagGroup();
        string root = STEMRoot() + ":Control:Scan:";
        tags.TagGroupSetTagAsLong(root + "PauseCount", self.LookupElement("scan-pause").DLGGetValue());
        tags.TagGroupSetTagAsLong(root + "ReadCount", self.LookupElement("scan-read").DLGGetValue());
        tags.TagGroupSetTagAsLong(root + "PositionsX", self.LookupElement("scan-x").DLGGetValue());
        tags.TagGroupSetTagAsLong(root + "Rows", self.LookupElement("scan-rows").DLGGetValue());
        tags.TagGroupSetTagAsLong(root + "Flyback", self.LookupElement("scan-flyback").DLGGetValue());
        tags.TagGroupSetTagAsBoolean(root + "FlushMemory", self.LookupElement("scan-flush").DLGGetValue());
    }

    void OnConfigureScan(object self) { self.StoreScan(); self.QueueCommand("configure_scan"); }
    void OnStartScan(object self) { self.QueueCommand("start_scan"); }
    void OnStopScan(object self) { self.QueueCommand("stop_scan"); }
    void OnAbortScan(object self) { self.QueueCommand("abort_scan"); }

    taggroup CreateStatusBox(object self)
    {
        taggroup items;
        taggroup box = DLGCreateBox("DAQ status", items);
        items.DLGAddElement(STEMLabeledField("DM engine", DLGCreateStringField("waiting", 28).DLGIdentifier("status-engine")));
        items.DLGAddElement(STEMLabeledField("Control", DLGCreateStringField("waiting", 28).DLGIdentifier("status-control")));
        items.DLGAddElement(STEMLabeledField("Acquisition", DLGCreateStringField("unknown", 28).DLGIdentifier("status-acquisition")));
        items.DLGAddElement(STEMLabeledField("Visualization", DLGCreateStringField("unknown", 28).DLGIdentifier("status-visualization")));
        items.DLGAddElement(STEMLabeledField("Burst", DLGCreateStringField("unknown", 28).DLGIdentifier("status-burst")));
        items.DLGAddElement(STEMLabeledField("Instrument", DLGCreateStringField("unknown", 28).DLGIdentifier("status-instrument")));
        items.DLGAddElement(STEMLabeledField("Message", DLGCreateStringField("", 28).DLGIdentifier("status-message")));
        items.DLGAddElement(DLGGroupItems(DLGCreatePushButton("Start acquisition", "OnStart"), DLGCreatePushButton("Stop acquisition", "OnStop"), DLGCreatePushButton("Refresh", "OnRefresh")));
        items.DLGAddElement(DLGCreatePushButton("Stop DM viewer and close", "OnStopViewer"));
        return box;
    }

    taggroup CreateVisualizationBox(object self)
    {
        string root = STEMRoot() + ":Control:Visualization:";
        taggroup items;
        taggroup box = DLGCreateBox("Thinned visualization", items);
        items.DLGAddElement(DLGCreateCheckBox("Publish products", STEMReadBoolean(root + "Publishing", 1)).DLGIdentifier("viz-publishing"));
        items.DLGAddElement(STEMLabeledField("Processing stage", STEMStageChoice("viz-stage", STEMReadString(root + "ProcessingStage", "corrected"))));
        items.DLGAddElement(STEMLabeledField("Total refresh Hz", DLGCreateRealField(STEMReadNumber(root + "RefreshHz", 1), 10, 3).DLGIdentifier("viz-rate")));
        items.DLGAddElement(STEMLabeledField("Representative frame", DLGCreateIntegerField(STEMReadNumber(root + "RepresentativeFrame", 64), 10).DLGIdentifier("viz-representative")));
        items.DLGAddElement(DLGGroupItems(DLGCreateCheckBox("Single frame", STEMReadBoolean(root + "IncludeRepresentative", 1)).DLGIdentifier("viz-include-representative"), DLGCreateCheckBox("128-frame sum", STEMReadBoolean(root + "IncludeSum", 1)).DLGIdentifier("viz-include-sum")));
        items.DLGAddElement(STEMLabeledField("ZLP threshold", DLGCreateRealField(STEMReadNumber(root + "ZLPThreshold", 0), 12, 3).DLGIdentifier("viz-zlp")));
        items.DLGAddElement(STEMLabeledField("CoreLoss threshold", DLGCreateRealField(STEMReadNumber(root + "CoreLossThreshold", 0), 12, 3).DLGIdentifier("viz-core")));
        items.DLGAddElement(DLGCreatePushButton("Apply visualization settings", "OnApplyVisualization"));
        return box;
    }

    taggroup CreateBurstBox(object self)
    {
        string root = STEMRoot() + ":Control:Burst:";
        taggroup items;
        taggroup box = DLGCreateBox("Controlled burst capture", items);
        items.DLGAddElement(STEMLabeledField("Processing stage", STEMStageChoice("burst-stage", STEMReadString(root + "ProcessingStage", "corrected"))));
        items.DLGAddElement(STEMLabeledField("File template", DLGCreateStringField(STEMReadString(root + "FilepathTemplate", "/data/stem_burst_rx{receiver}_{capture}_{stage}.h5"), 45).DLGIdentifier("burst-file")));
        items.DLGAddElement(STEMLabeledField("Dataset", DLGCreateStringField(STEMReadString(root + "DatasetName", "/frames"), 24).DLGIdentifier("burst-dataset")));
        items.DLGAddElement(STEMLabeledField("Buckets per capture", DLGCreateIntegerField(STEMReadNumber(root + "BucketsPerCapture", 1), 10).DLGIdentifier("burst-buckets")));
        items.DLGAddElement(STEMLabeledField("Captures per arm (0 unlimited)", DLGCreateIntegerField(STEMReadNumber(root + "CaptureCount", 1), 10).DLGIdentifier("burst-count")));
        items.DLGAddElement(DLGGroupItems(DLGCreateCheckBox("Re-arm after write", STEMReadBoolean(root + "RearmAfterWrite", 1)).DLGIdentifier("burst-rearm"), DLGCreateCheckBox("Require complete buckets", STEMReadBoolean(root + "StrictComplete", 0)).DLGIdentifier("burst-strict")));
        items.DLGAddElement(STEMLabeledField("ZLP threshold", DLGCreateRealField(STEMReadNumber(root + "ZLPThreshold", 0), 12, 3).DLGIdentifier("burst-zlp")));
        items.DLGAddElement(STEMLabeledField("CoreLoss threshold", DLGCreateRealField(STEMReadNumber(root + "CoreLossThreshold", 0), 12, 3).DLGIdentifier("burst-core")));
        items.DLGAddElement(DLGGroupItems(DLGCreatePushButton("Apply settings", "OnConfigureBurst"), DLGCreatePushButton("Apply and arm", "OnArmBurst")));
        items.DLGAddElement(DLGGroupItems(DLGCreatePushButton("Disarm after capture", "OnDisarmBurst"), DLGCreatePushButton("Abort capture", "OnAbortBurst")));
        return box;
    }

    taggroup CreateCameraBox(object self)
    {
        taggroup items;
        taggroup box = DLGCreateBox("Camera head and detector links", items);
        items.DLGAddElement(STEMLabeledField("Service", DLGCreateStringField("offline", 28).DLGIdentifier("camera-service")));
        items.DLGAddElement(STEMLabeledField("Mode", DLGCreateStringField("unknown", 28).DLGIdentifier("camera-mode")));
        items.DLGAddElement(STEMLabeledField("Power", DLGCreateStringField("unknown", 28).DLGIdentifier("camera-power")));
        items.DLGAddElement(STEMLabeledField("Insertion", DLGCreateStringField("unknown", 28).DLGIdentifier("camera-insertion")));
        items.DLGAddElement(STEMLabeledField("Temperature C", DLGCreateRealField(0, 12, 2).DLGIdentifier("camera-temperature")));
        items.DLGAddElement(STEMLabeledField("Target C", DLGCreateRealField(0, 12, 2).DLGIdentifier("camera-target")));
        items.DLGAddElement(DLGGroupItems(DLGCreatePushButton("Power up", "OnCameraPowerUp"), DLGCreatePushButton("Power down", "OnCameraPowerDown")));
        items.DLGAddElement(DLGGroupItems(DLGCreatePushButton("Insert", "OnCameraInsert"), DLGCreatePushButton("Retract", "OnCameraRetract")));
        items.DLGAddElement(DLGGroupItems(DLGCreatePushButton("Read temperature", "OnCameraTemperature"), DLGCreatePushButton("Read biases", "OnCameraBiases")));
        items.DLGAddElement(STEMLabeledField("Detector links", DLGCreateStringField("unknown", 28).DLGIdentifier("detector-links")));
        items.DLGAddElement(STEMLabeledField("Synchronized", DLGCreateStringField("no", 28).DLGIdentifier("detector-sync")));
        items.DLGAddElement(STEMLabeledField("Aligned", DLGCreateStringField("no", 28).DLGIdentifier("detector-aligned")));
        items.DLGAddElement(DLGGroupItems(DLGCreatePushButton("Read links", "OnDetectorLinks"), DLGCreatePushButton("Resync", "OnDetectorResync"), DLGCreatePushButton("Auto-align", "OnDetectorAutoAlign")));
        items.DLGAddElement(STEMLabeledField("Operation", DLGCreateStringField("", 28).DLGIdentifier("operation-name")));
        items.DLGAddElement(STEMLabeledField("State", DLGCreateStringField("idle", 28).DLGIdentifier("operation-state")));
        items.DLGAddElement(STEMLabeledField("Current step", DLGCreateStringField("", 28).DLGIdentifier("operation-step")));
        items.DLGAddElement(STEMLabeledField("Steps complete", DLGCreateIntegerField(0, 8).DLGIdentifier("operation-completed")));
        items.DLGAddElement(STEMLabeledField("Total steps", DLGCreateIntegerField(0, 8).DLGIdentifier("operation-total")));
        items.DLGAddElement(STEMLabeledField("Error", DLGCreateStringField("", 28).DLGIdentifier("operation-error")));
        items.DLGAddElement(DLGCreatePushButton("Refresh instrument state", "OnInstrumentRefresh"));
        return box;
    }

    taggroup CreateScanBox(object self)
    {
        string root = STEMRoot() + ":Control:Scan:";
        taggroup items;
        taggroup box = DLGCreateBox("Mock scan configuration and control", items);
        items.DLGAddElement(STEMLabeledField("State", DLGCreateStringField("idle", 28).DLGIdentifier("scan-state")));
        items.DLGAddElement(STEMLabeledField("Scan number", DLGCreateIntegerField(0, 10).DLGIdentifier("scan-number")));
        items.DLGAddElement(STEMLabeledField("Expected frames", DLGCreateIntegerField(0, 12).DLGIdentifier("scan-expected")));
        items.DLGAddElement(STEMLabeledField("Received frames", DLGCreateIntegerField(0, 12).DLGIdentifier("scan-received")));
        items.DLGAddElement(STEMLabeledField("Pause count", DLGCreateIntegerField(STEMReadNumber(root + "PauseCount", 200), 10).DLGIdentifier("scan-pause")));
        items.DLGAddElement(STEMLabeledField("Read count", DLGCreateIntegerField(STEMReadNumber(root + "ReadCount", 1), 10).DLGIdentifier("scan-read")));
        items.DLGAddElement(STEMLabeledField("X positions", DLGCreateIntegerField(STEMReadNumber(root + "PositionsX", 1), 10).DLGIdentifier("scan-x")));
        items.DLGAddElement(STEMLabeledField("Rows", DLGCreateIntegerField(STEMReadNumber(root + "Rows", 1), 10).DLGIdentifier("scan-rows")));
        items.DLGAddElement(STEMLabeledField("Flyback", DLGCreateIntegerField(STEMReadNumber(root + "Flyback", 100), 10).DLGIdentifier("scan-flyback")));
        items.DLGAddElement(DLGCreateCheckBox("Flush detector memory", STEMReadBoolean(root + "FlushMemory", 1)).DLGIdentifier("scan-flush"));
        items.DLGAddElement(DLGCreatePushButton("Apply scan configuration", "OnConfigureScan"));
        items.DLGAddElement(DLGGroupItems(DLGCreatePushButton("Start scan", "OnStartScan"), DLGCreatePushButton("Stop scan", "OnStopScan"), DLGCreatePushButton("Abort scan", "OnAbortScan")));
        return box;
    }

    taggroup CreateDialog(object self)
    {
        taggroup items;
        taggroup dialog = DLGCreateDialog("STEM DAQ Control", items);
        taggroup tabs = DLGCreateTabList(0);
        taggroup statusTab = tabs.DLGAddTab("Status");
        taggroup visualizationTab = tabs.DLGAddTab("Visualization");
        taggroup burstTab = tabs.DLGAddTab("Burst");
        taggroup cameraTab = tabs.DLGAddTab("Camera");
        taggroup scanTab = tabs.DLGAddTab("Scan");
        statusTab.DLGAddElement(self.CreateStatusBox());
        visualizationTab.DLGAddElement(self.CreateVisualizationBox());
        burstTab.DLGAddElement(self.CreateBurstBox());
        cameraTab.DLGAddElement(self.CreateCameraBox());
        scanTab.DLGAddElement(self.CreateScanBox());
        taggroup wrapper = DLGCreateGroup();
        wrapper.DLGAddElement(tabs);
        items.DLGAddElement(wrapper);
        return dialog;
    }

    object Init(object self)
    {
        self.super.Init(self.CreateDialog());
        return self;
    }
}

{
    object palette = Alloc(STEMDAQControlPalette).Init();
    palette.Display("STEM DAQ Control");
}
