import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_NAMES = new Set(["StereoVideoSource"]);
const VIDEO_WIDGETS = [
  { name: "source_video", label: "source video" },
];

function makeUploadHandler(node, fileWidget) {
  return () => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".mp4,.webm,.mkv,.gif,.mov,.avi,.m4v";

    input.addEventListener("change", async () => {
      if (!input.files?.length) {
        return;
      }

      const file = input.files[0];
      const formData = new FormData();
      formData.append("image", file);
      formData.append("overwrite", "true");
      formData.append("type", "input");

      try {
        const response = await api.fetchApi("/upload/image", {
          method: "POST",
          body: formData,
        });

        if (response.status !== 200) {
          alert(`Error uploading file: ${response.statusText}`);
          return;
        }

        const data = await response.json();
        if (!data.name) {
          alert("Upload finished but no file name was returned.");
          return;
        }

        fileWidget.value = data.name;
        if (typeof fileWidget.callback === "function") {
          fileWidget.callback(data.name);
        }
        app.graph.setDirtyCanvas(true, true);
      } catch (error) {
        alert(`Error uploading file: ${error.message}`);
      }
    });

    input.click();
  };
}

app.registerExtension({
  name: "StereoLongVideo.Upload",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODE_NAMES.has(nodeData.name)) {
      return;
    }

    const original = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = original ? original.apply(this, arguments) : undefined;

      for (const widgetConfig of VIDEO_WIDGETS) {
        const fileWidget = this.widgets?.find((widget) => widget.name === widgetConfig.name);
        if (!fileWidget) {
          continue;
        }

        const buttonName = `Upload ${widgetConfig.label}`;
        const existing = this.widgets.find((widget) => widget.name === buttonName);
        if (existing) {
          continue;
        }

        this.addWidget("button", buttonName, "upload", makeUploadHandler(this, fileWidget));
      }

      return result;
    };
  },
});
