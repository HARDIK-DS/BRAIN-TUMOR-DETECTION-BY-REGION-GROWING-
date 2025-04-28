modules = ["python-3.11"]

[nix]
channel = "stable-24_05"
packages = ["cairo", "ffmpeg-full", "freetype", "ghostscript", "gobject-introspection", "gtk3", "itk", "libGL", "libGLU", "pkg-config", "qhull", "swig4", "tcl", "tk"]

[deployment]
deploymentTarget = "autoscale"
run = ["streamlit", "run", "app.py", "--server.port", "5000"]

[workflows]
runButton = "Project"

[[workflows.workflow]]
name = "Project"
mode = "parallel"
author = "agent"

[[workflows.workflow.tasks]]
task = "workflow.run"
args = "BrainTumorDetection"

[[workflows.workflow]]
name = "BrainTumorDetection"
author = "agent"

[workflows.workflow.metadata]
agentRequireRestartOnSave = false

[[workflows.workflow.tasks]]
task = "packager.installForAll"

[[workflows.workflow.tasks]]
task = "shell.exec"
args = "streamlit run app.py --server.port 5000"
waitForPort = 5000

[[ports]]
localPort = 5000
externalPort = 80
