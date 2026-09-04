import os
from roboflow import Roboflow

# 1. Initialize with your API Key
rf = Roboflow(api_key="Cw6CEN9V8wqQZ1if6DiA")

# 2. Define workspaces to download from
workspaces_to_fetch = ["robotics-lab-1", "mtsu"]

# 3. Create a directory to store all datasets
base_download_path = "./data/training-data/downloaded/"
os.makedirs(base_download_path, exist_ok=True)

for ws_name in workspaces_to_fetch:
    print(f"\n--- Fetching projects for workspace: {ws_name} ---")
    try:
        workspace = rf.workspace(ws_name)
        projects = workspace.projects()
        
        for project_meta in projects:
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
            except Exception as e:
                print(f"Could not download project {project_id}: {e}")
                
    except Exception as e:
        print(f"Could not access workspace {ws_name}: {e}")

print("\nDone!")
