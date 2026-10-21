from roboflow import Roboflow
rf = Roboflow(api_key="sOm4OTW6Gfa5yXCBC0fx")
project = rf.workspace("izzulh-ocslp").project("pothole-6q3m8")
version = project.version(1)
dataset = version.download("yolov8")