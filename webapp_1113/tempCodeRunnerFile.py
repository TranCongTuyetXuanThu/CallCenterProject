# def extract_mfcc(wav_file_name):
#     y, sr = librosa.load(wav_file_name)
#     mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
#     return mfccs

# def predict(model, wav_filepath):
#     emotions = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'}
#     test_point = extract_mfcc(wav_filepath)
#     test_point = np.reshape(test_point, newshape=(1, 40, 1))
#     predictions = model.predict(test_point)
#     return emotions[np.argmax(predictions[0]) + 1]
# def predict_1_sample(file_sample, model, preprocesser):
#     file_sample, tensor = preprocesser.complete_preprocessing(file_sample)
#     output_logit, output_softmax = model(tensor)
#     output_softmax = torch.argmax(output_softmax, dim=1)
#     final_output = max(set(output_softmax.tolist()), key=output_softmax.tolist().count)
#     emotion_dict = {0: 'positive',
#                     1: 'neutral',
#                     2: 'negative'}
#     label = emotion_dict[final_output]
#     return final_output, label

# import wave
