By Murtadha Hssayeni, mhssayeni2017@fau.edu 8/13/2018

Dataset name: ct-ich
Authors: Murtadha D. Hssayeni, M.S., Muayad S. Croock, Ph.D., Aymen Al-Ani, Ph.D., Hassan Falah Al-khafaji, M.D., Zakaria A. Yahya, M.D., and Behnaz Ghoraani, Ph.D.

A retrospective study was designed to collect head CT scans of subjects with TBI and it was approved by the research and ethics board in the Iraqi Ministry of Health, Babil Office (approval #1369). The inclusion criteria were any subject who was admitted to the hospital emergency unit with a TBI, and a CT scan was performed to him/her. CT scans were collected between February and August 2018 from Al Hilla Teaching Hospital, Iraq. Sensitive information for each patient was anonymized, and the subjects' faces were de-identified in the CT scans.

A dataset of 82 CT scans was collected, including 36 scans for patients diagnosed with intracranial hemorrhage with the following types: Intraventricular, Intraparenchymal, Subarachnoid, Epidural and Subdural. Each CT scan for each patient includes about 30 slices with 5 mm slice-thickness. The mean and std of patients' age were 27.8 and 19.5, respectively. 46 of the patients were males and 36 of them were females. Each slice of the non-contrast CT scans was by two radiologists who recorded hemorrhage types if hemorrhage occurred or if a fracture occurred. The radiologists also delineated the ICH regions in each slice. There was a consensus between the radiologists. Radiologists did not have access to clinical history of the patients.

During data collection, syngo by Siemens Medical Solutions was first used to read the CT DICOM files and save two videos (avi format) using brain (level=40, width=120) and bone (level=700, width=3200) windows, respectively. Second, a custom tool was implemented in Matlab and used to read the avi files and perform the annotation. Also, the generated masks were mapped and saved to NIfTI files for 75 subjects. 

Files and folders in the dataset are:
	patient_demographics.csv contains the patient #, age and gender and the labels (the ICH subtypes if ICH was diagnosed, and a skull fracture if it was diagnosed) for each CT scan.
	hemorrhage_diagnosis_raw_ct.csv contains the patient #, slice # and the labels (the ICH subtypes if ICH was diagnosed, and a skull fracture if it was diagnosed) for each slice in the NIfTI CT scans.
	ct_scans folder contains the NIfTI scans for the patients in the patient demographics file except for subject #59 to 65 which are missing. The names of the NIfTI mask files match the patients numbers in the patient demographics file (patient_demographics.csv and hemorrhage_diagnosis_raw_ct.csv).
	masks folder contains the ICH segmentation of each of the CT slices in NIfTI file format. The names of the NIfTI mask files match the patients numbers in the patient demographics file.

split_raw_data.py is provided to load the NIfTI CT scans and window them using a brain window, and also to load the NIfTI masks. An environment file is provided, ct_ich.yml, which specifies the virtual environment that was used to execute the code. The virtual environment can be recreated using conda as follows:
conda env create -f ct_ich.yml
This will create the ct_ich python virtual environment for running the code.

In this release, only the CT scans in the NIfTI format were included, and the subjects' faces in these CT scans were de-identified by blurring them and adding random noise. 