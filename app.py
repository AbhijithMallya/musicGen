from transformers import pipeline
import scipy
import pickle


synthesiser = pickle.load(open('model.pkl','rb'))


music = synthesiser("Indian Wedding music", forward_params={"do_sample": True})

scipy.io.wavfile.write("musicgen_out.wav", rate=music["sampling_rate"], data=music["audio"])