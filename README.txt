The Emotion Detection project aims to leverage machine learning techniques in order to classify voice recordings from the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) into one of three different emotions: happy, sad, and angry. By doing so, our team hopes to improve the customer service space by better allocating resources based on the emotions detected in a customer's voice. 

As per the description on the RAVDESS website, the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) contains 7356 files (total size: 24.8 GB). The database contains 24 professional actors (12 female, 12 male), vocalizing two lexically-matched statements in a neutral North American accent. Speech includes calm, happy, sad, angry, fearful, surprise, and disgust expressions, and song contains calm, happy, sad, angry, and fearful emotions. Each expression is produced at two levels of emotional intensity (normal, strong), with an additional neutral expression. All conditions are available in three modality formats: Audio-only (16bit, 48kHz .wav), Audio-Video (720p H.264, AAC 48kHz, .mp4), and Video-only (no sound). For our project, we used the audio-only speech recordings pertaining to happy, sad, and angry expressions.

The first step in this process is to collect .wav files from the RAVDESS database and store them in a folder according to their labelled emotions. Next, we use the Librosa python library to apply a standard Fourier transform and then convert each file into a spectrogram image. Additionally, we also convert the amplitude of each image to decibels as a normalization step. After using Matplotlib to save each image, we then use OpenCV to read in each image as an array. Subsequently, we train an AlexNet model on our set of spectrogram images and corresponding emotional labels. Lastly, once the AlexNet model is trained, it will be able to perform classification on a new voice recording and present the results in a web application. 

AlexNet is a well-known CNN architecture that has proven to be effective for image classification tasks. It consists of five convolutional layers, followed by 3 fully connected layers, and ends with an output layer. Moreover, the first, second, and fifth convolutional layers consist of MaxPooling layers with the purpose of downsampling the data while preserving its key information. Overall, this serves to increasing training speed without having too much of an impact on the model’s ability to retain information.  

Packages required:
- TensorFlow
- Matplotlib
- scikit-learn
- OpenCV
- Librosa
- Voila

Steps to run the code:
1. Download the training set by visiting https://zenodo.org/record/1188976 and choosing the files corresponding to a modality of audio-only, vocal channel of speech, and emotions of happy, sad and angry. For best results, choose a mixture of male and female actors.
2. Group the downloaded recordings into one folder
3. Update audio_fpath in sound_to_image_converter.ipynb such that it reflects the path of the folder containing the downloaded recordings
4. Run sound_to_image_converter.ipynb
