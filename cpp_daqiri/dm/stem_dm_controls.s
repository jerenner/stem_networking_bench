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
        self.DLGValue("status-message", STEMReadString(base + "Message", ""));
    }

    void ReloadSettings(object self)
    {
        string visual = STEMRoot() + ":Control:Visualization:";
        string burst = STEMRoot() + ":Control:Burst:";
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
        self.RefreshStatus();
    }

    void OnStart(object self) { self.QueueCommand("start_acquisition"); }
    void OnStop(object self) { self.QueueCommand("stop_acquisition"); }
    void OnRefresh(object self) { self.ReloadSettings(); }

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

    taggroup CreateStatusBox(object self)
    {
        taggroup items;
        taggroup box = DLGCreateBox("DAQ status", items);
        items.DLGAddElement(STEMLabeledField("DM engine", DLGCreateStringField("waiting", 28).DLGIdentifier("status-engine")));
        items.DLGAddElement(STEMLabeledField("Control", DLGCreateStringField("waiting", 28).DLGIdentifier("status-control")));
        items.DLGAddElement(STEMLabeledField("Acquisition", DLGCreateStringField("unknown", 28).DLGIdentifier("status-acquisition")));
        items.DLGAddElement(STEMLabeledField("Visualization", DLGCreateStringField("unknown", 28).DLGIdentifier("status-visualization")));
        items.DLGAddElement(STEMLabeledField("Burst", DLGCreateStringField("unknown", 28).DLGIdentifier("status-burst")));
        items.DLGAddElement(STEMLabeledField("Message", DLGCreateStringField("", 28).DLGIdentifier("status-message")));
        items.DLGAddElement(DLGGroupItems(DLGCreatePushButton("Start acquisition", "OnStart"), DLGCreatePushButton("Stop acquisition", "OnStop"), DLGCreatePushButton("Refresh", "OnRefresh")));
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

    taggroup CreateDialog(object self)
    {
        taggroup items;
        taggroup dialog = DLGCreateDialog("STEM DAQ Control", items);
        items.DLGAddElement(self.CreateStatusBox());
        items.DLGAddElement(self.CreateVisualizationBox());
        items.DLGAddElement(self.CreateBurstBox());
        return dialog;
    }

    object Init(object self)
    {
        self.super.Init(self.CreateDialog());
        return self;
    }
}

object gSTEMDAQControlPalette = Alloc(STEMDAQControlPalette).Init();
gSTEMDAQControlPalette.Display("STEM DAQ Control");
