# importing the libraries
import streamlit as st
from autolabel import LabelingAgent
import json
import os

# Path to your dataset file
DATASET_FILE = "dataset.csv"

# Create the "users_data" folder to save the data that users have upload, later we can saves the files in the sql, or simply remove them after sometime
if not os.path.exists("users_data"):
    os.makedirs("users_data")


st.title("Autolabel")
st.write("Welcome to Autolabel!")

# upload JSON file
json_file = st.file_uploader("Upload JSON file", type="json")
json_config = None

if json_file is not None:
    json_config = json.load(json_file)

# manually enter JSON file if there is no file saved
json_config_input = st.text_area("Enter JSON configuration", height=200)
manual_json_config = None

# upload the csv file
dataset_file = st.file_uploader("Upload dataset file (CSV)", type="csv")

if st.button("Process"):
    if json_config is not None:
        agent = LabelingAgent(config=json_config)
    elif json_config_input:
        manual_json_config = json.loads(json_config_input)
        agent = LabelingAgent(config=manual_json_config)
    else:
        st.warning(
            "Please upload a JSON configuration file or enter JSON configuration."
        )

    if agent and dataset_file is not None:
        # Save the uploaded files in the "users_data" folder
        json_filename = (
            os.path.join("users_data", json_file.name)
            if json_file is not None
            else None
        )
        dataset_filename = os.path.join("users_data", dataset_file.name)

        if json_filename:
            with open(json_filename, "wb") as f:
                f.write(json_file.read())

        with open(dataset_filename, "wb") as f:
            f.write(dataset_file.read())

        agent.plan(dataset_filename)

        labels, output_df, metrics = agent.run(dataset_filename)

        st.subheader("Labeled Examples")
        st.dataframe(output_df)

        # Option to download the labeled dataset
        if st.button("Download Labeled Dataset"):
            labeled_dataset_filename = os.path.join("users_data", "labeled_dataset.csv")
            output_df.to_csv(labeled_dataset_filename, index=False)
            st.success("Labeled dataset downloaded successfully!")
