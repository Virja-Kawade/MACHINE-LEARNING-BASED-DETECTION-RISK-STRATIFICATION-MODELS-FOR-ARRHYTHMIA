# MACHINE-LEARNING-BASED-DETECTION-RISK-STRATIFICATION-MODELS-FOR-ARRHYTHMIA
Worldwide, heart disease continues to be the primary cause of death, necessitating the creation of reliable predictive technologies to identify people who are at
high risk. Machine learning algorithms have recently become effective tools for predicting cardiac disease, taking use of their capacity to examine intricate patterns
within big datasets. With a focus on their methodology, performance, and clinical consequences, this in-depth review intends to evaluate the state-of-the-art in machine learning-based prediction models for heart disease risk assessment.

Arrhythmia is one such cardiovascular disease that can go undetected if not monitored continuously. In this study, a novel method for the detection of irregular
heartbeats, is proposed that willl help in the diagnosis of Arrhythmia. A deep learning based approach is carried out by making use of a 1D Convolutional Neural Network, that is trained on the MIT BIH Arrhythmia Database. In the final prediction task, heartbeat signal is classified into one of the two classes: Normal and Arrhythmia. The accuracy obtained for the 1D CNN is 0.985.

This project has been inspired by the work of Prof. Edward Mina in his study "The Application of Deep Convolutional Networks for the Classification of ECG Signal"
https://github.com/eddymina/ECG_Classification_Pytorch/tree/master

# METHODOLOGY

1) Normalizing signals
The raw signal data acuired from PhysioNet ATM’s MIT BIH Arrhythmia Database was first normalized from 0 to 1 using normalization equation as follows –

![image](https://github.com/Virja-Kawade/MACHINE-LEARNING-BASED-DETECTION-RISK-STRATIFICATION-MODELS-FOR-ARRHYTHMIA/assets/71089824/62e0abf4-d7db-4844-890b-831e1dd2810b)

2) Filtering the signals
A moving average filter was applied afterwards. A typical linear filter used in signal processing is the Butterworth lowpass filter, which attenuates high-frequency components while allowing low-frequency components to pass through. Its smooth frequency response and lack of passband ripples make it a popular choice. The family of infinite impulse response (IIR) filters includes the Butterworth filter. In order to keep the required low-frequency components of a signal while removing or attenuating unwanted high-frequency noise or interference is the main goal of utilizing a Butterworth lowpass filter. This is especially helpful when the signal of interest only exists within a narrow frequency range and other frequency components could contribute noise or artifacts that could impair the accuracy or quality of the signal analysis.

3) Applying Heartbeat Isolation Algorithm
The isolation algorithm based on R wave peak finder was employed to find Peaks. The specific one used here is the Christov Segmenter.

4) Labelling and Resolving Class Imbalance
The MIT BIH Arrhythmia Database has pre-labelled peaks, but in order to overcome the limitation of class imbalance, the existing labels are further grouped within
each other. As a result, the various classes are grouped into two major classes :
0 : Normal
1 : Arrhythmia

5) Resampling and Splitting the Data
The data is zero padded for making all the samples even and resampled from 360 Hz to 125 Hz to reduce the amount of data.The data is then split into training, testing and validation sets. It is significant to notice that each instance of the training batch comprised an equal number of data points for each class. Using a weighted sample technique, where each class weight was equal to the reciprocal of its count, this was accomplished. However, only the training data were used for this evaluation; the validation and test sets used the genuine distribution. The testing data was isolated randomly for each class rather than generically throughout the full data set because not all patients had access to all classes.

6) Training the CNN
When analyzing features from shorter (fixed-length) chunks of the entire data set and if the feature's placement inside the segment is not very relevant, a 1D CNN is particularly effective[6]. The model essentially consists of four blocks, each of which has two convolutions with a kernel size of 32 by 5 by 1, followed by a final max pooling layer with a kernel size of 5 and stride of 2. The convoluted output is then fed into a dense, completely connected series of layers, with the first dense layer being the only layer to have a ReLU activation function. The Log SoftMax function is used to activate the dense layer output in the end, which is 5 classes in number


# Defining Threshold for Arrhythmia Signals in ECG

The picture represents the annotation file as represented in MIT-BIH Arrhythmia Dataset. Here normal heartbeat rates are annotated as “N” and instances of Arrhythmia are annotated as “A”. Arrhythmia occurs when either of the following cases occur: Atrial premature beat, Aberrated atrial premature beat, Nodal (junctional) premature beat, Supraventricular premature beat, or Premature ventricular contraction. In Table 5 and Atrial Premature Beat is shown where the heart beat occurs before the expected time interval (i.e., it occurs after 235 Hz whereas other heartbeats occurs within the expected timeframe of 280 – 300 Hz). This abberation in the frequency of occurrence of a hearbeat does not cause the next heartbeat to occur earlier, rather it occurs at its expected time frame, thus causing a lag in the frequency. This is the anatomy behind the occurrence of Arrhythmia and the 1D CNN detects this kind of discrepancy.

![image](https://github.com/Virja-Kawade/MACHINE-LEARNING-BASED-DETECTION-RISK-STRATIFICATION-MODELS-FOR-ARRHYTHMIA/assets/71089824/8eb9b95d-f045-4521-b57c-6b0ceb6ee97e)

# Dataset

The MIT-BIH Arrhythmia Database will provide the data for the experiment. 48 patients at Beth-Israel Hospital have 30 minute (360 samples/sec) ECG recordings available in the database below, which contains information going all the way back to 1975. From a pool of 4000 24-hour ambulatory ECG recordings made by a mixed group of inpatients (approximately 60%) and outpatients (about 40%), 23 recordings are chosen
30 at random. To incorporate less frequent but clinically important arrhythmias that would not be well represented in a small random sample, the remaining 25 recordings from the same set were chosen

Link to the MIT-BIH Dataset - https://physionet.org/physiobank/database/html/mitdbdir/mitdbdir.htm

Link to PhysioBank ATM - https://archive.physionet.org/cgi-bin/atm/ATM

![image](https://github.com/Virja-Kawade/MACHINE-LEARNING-BASED-DETECTION-RISK-STRATIFICATION-MODELS-FOR-ARRHYTHMIA/assets/71089824/9859bce9-ac7d-4fc8-9e29-2759e1798971)
Fig. Six types of signal annotations classified into two classes for ease of understanding in the MIT BIH Arrhythmia Database

![image](https://github.com/Virja-Kawade/MACHINE-LEARNING-BASED-DETECTION-RISK-STRATIFICATION-MODELS-FOR-ARRHYTHMIA/assets/71089824/b17e5847-408c-475d-9238-4174190db78a)
Fig. Sample 10-second ECG data of patient collected from MIT BIH Arrhythmia Database showing normal beats annotated as “.” and a single arrhythmia beat annotated as “A” (Source – PhysioNet ATM “MIT BIH Arrhythmia
Database”)

# Results

The accuracy obtained on the test dataset is 0.985. The precision, recall, F1 score and support values that were recorded are shown in table.

![image](https://github.com/Virja-Kawade/MACHINE-LEARNING-BASED-DETECTION-RISK-STRATIFICATION-MODELS-FOR-ARRHYTHMIA/assets/71089824/a9dd4a88-db18-4ecc-9dd8-a0293cadc7de)
Fig. Normalized confusion matrix for on the test set of MIT BIH Arrhythmia Database

![image](https://github.com/Virja-Kawade/MACHINE-LEARNING-BASED-DETECTION-RISK-STRATIFICATION-MODELS-FOR-ARRHYTHMIA/assets/71089824/0d9614bc-aba6-47b7-8fe0-86944bf30743)
Fig. Precision, recall, F1-score and support values of 1-D CNN on MIT BIH Arrhythmia database





