import os
from roboflow import Roboflow

# 1. Initialize with your API Key
# You can find this in your Roboflow settings (https://app.roboflow.com/settings/api)
rf = Roboflow(api_key="JvguSIcRS5w7c7RFtxlf")

# 2. Access your workspace
# If you leave it blank, it defaults to your primary workspace
workspace = rf.workspace("robotics-lab-1")

# 3. Create a directory to store all datasets
base_download_path = "./data/training-data/downloaded/"
os.makedirs(base_download_path, exist_ok=True)

# 4. Iterate through all projects in the workspace
print(f"Fetching projects for workspace: {workspace.name}")
projects = workspace.projects()

print(projects)

for project_meta in projects:
    # Extract the actual ID/Slug from the dictionary
    # Depending on SDK version, it's usually 'id' or 'name'
    project_id = project_meta.split("/")[-1] if "/" in project_meta else project_meta
    
    print(f"Loading project: {project_id}")
    
    try:
        project = workspace.project(project_id)
        versions = project.versions()
        
        if versions:
            latest_version = versions[0].version
            print(f"Downloading {project_id} v{latest_version}...")
            
            project.version(latest_version).download(
                model_format="yolov5", 
                location=os.path.join(base_download_path, project_id)
            )

            os.path.join(base_download_path, project_id)

    except Exception as e:
        print(f"Could not download {project_id}: {e}")

print("Done!")