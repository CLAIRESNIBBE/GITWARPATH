
import neptune.new as neptune
run = neptune.init(project='clairesnibbe/XGBRegressor',
api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyY2QzMDNmMi1mMmJjLTQ1ZjEtOWVjMi0yMzU3MTYzZjcxOWUifQ==")
                 # Track metadata and hyperparameters of your Run
run["JIRA"] = "NPT-952"
run["algorithm"] = "ConvNet"

params = {"batch_size": 64, "dropout": 0.2, "learning_rate": 0.001, "optimizer": "Adam"}
run["parameters"] = params



# Track the training process by logging your training metrics
for epoch in range(100):
    run["train/accuracy"].log(epoch * 0.6)
    run["train/loss"].log(epoch * 0.4)
# Log the final results
run["f1_score"] = 0.66

# Stop logging to your Run
run.stop()
Step 3: Add your credentials
To log metadata to Neptune you need to pass your credentials to the neptune.init() method.
run = neptune.init(
    project="YOUR_WORKSPACE/YOUR_PROJECT", api_token="YOUR_API_TOKEN"
)  # your credentials
project
The project argument has the format workspace_name/project_name
To find it:
Go to the Neptune UI
Go to your project
Open Settings > Properties
Copy the project name
api_token
To find it:
Go to the Neptune UI
Open the User menu toggle in the upper right
Click Get Your API token
Copy your API token
or get your API token directly from .
How to find your Neptune API token
For example:
run = neptune.init(
    project="funky_steve/timeseries",
    api_token="eyJhcGlfYW908fsdf23f940jiri0bn3085gh03riv03irn",
)
Step 4: Run your script and explore the results
Now that you have your script ready, run it from your terminal, Jupyter Lab, or other environments.
python hello_world.py
Click on the link in the terminal or notebook or go directly to the Neptune app.
The link should look like this one:
​
See metrics you logged in All Metadata, and Charts sections or go to the Monitoring section to see the hardware consumption.
Conclusion
You’ve learned how to:
Install neptune-client,
Connect Neptune to your Python script and create a run,
Log metrics to Neptune,
Explore your metrics in All metadata and Charts sections,
See hardware consumption during the run execution in the Monitoring section.
​

​
​

​
​

​
What’s next?