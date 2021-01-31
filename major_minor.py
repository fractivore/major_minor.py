import sys, os, json
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile # For reading/writing wav files
from IPython import embed # For debugging

#commandLineOptions =
if "--file" in sys.argv:
    input_wav = sys.argv[sys.argv.index("--file") + 1]

if "--range" in sys.argv:
    input_range = int(sys.argv[sys.argv.index("--range") + 1])
else:
    input_range = 100
if "--savePath" in sys.argv:
    save_path = int(sys.argv[sys.argv.index("--savePath") + 1])
else:
    save_path = "."
# Read the input wav file
[SAMPLE_RATE, audio] = scipy.io.wavfile.read(input_wav)
print("Sample Rate:", SAMPLE_RATE)
left_audio_data = np.array([data_point[0] for data_point in audio])
right_audio_data = np.array([data_point[1] for data_point in audio])

# static variables
HALF_STEP_RATIO = 1.0594630944
BUFFER_SIZE = 2*2*2*4096
temporal_resolution = BUFFER_SIZE/SAMPLE_RATE
print("Buffer size (fft bands):", BUFFER_SIZE)
FREQUENCY_RESOLUTION = SAMPLE_RATE/BUFFER_SIZE;
naked_filename_in = os.path.basename(input_wav)
filename_out = save_path +"/" + os.path.splitext(naked_filename_in)[0] + "-converted" + os.path.splitext(naked_filename_in)[1]

starting_freq_spectrum = np.fft.fft(left_audio_data)

def main():
    transposition_definition_dict = import_cache_file("scale_mode_transpositions.json")
    key_in = input('Please enter the letter of the key of the song (e.g. c or f#):')
    mode_in = input('Please enter the current mode of the song (e.g. major):')
    mode_out = input('Please enter the target mode (e.g. minor):')
    #get the sequence of note transformations from the dictionary using user input
    mode_conversion = transposition_definition_dict[key_in][mode_in][mode_out]
    print("mode_conversion:", mode_conversion)
    #reformat the data from the json required format
    mode_conversion = {int(k):v for k,v in mode_conversion.items()}
    #convert the data using the transformations obtained from the definition dictionary
    wav_data_out = convert_key(mode_conversion)
    #At the very end, write all our data into a nice new wav file:
    converted_wav_data = np.array(wav_data_out)
    #plot the frequency spectrum of the finished product vs the starting file for comparison:
    end_freq_spectrum = np.fft.fft(wav_data_out)
    scipy.io.wavfile.write(filename_out, SAMPLE_RATE, converted_wav_data.astype(np.int16))
    plot_frequency_spectrum(starting_freq_spectrum, end_freq_spectrum,input_range)

def pitchRecognize(FFTIndexIn):
    "Returns the pitch corresponding to a given index from the output of an FFT, based on the SAMPLE_RATE and BUFFER_SIZE parameters in this Visualizer instance."
    #Some convenience variables, for index in the frequency spectrum
    #and for the calculated frequency resolution:
    ind = FFTIndexIn
    FREQUENCY_RESOLUTION = SAMPLE_RATE/BUFFER_SIZE;
    sensitivity_coefficient = 1
    E = 1
    #p represents an exponent. We span the following range
    #exponentially because that's how notes relate to frequencies.
    for p in [*range(10)]:
        if ((16.35*pow(2,p))-pow(2,p-1)*.98*sensitivity_coefficient < ind*FREQUENCY_RESOLUTION and ind*FREQUENCY_RESOLUTION  < (16.35*pow(2,p))+pow(2,p-1)*.98*sensitivity_coefficient):
            return 0 #Note = C

        elif (17.32*pow(2,p)-pow(2,p-1)*1.03*sensitivity_coefficient < ind*FREQUENCY_RESOLUTION and ind*FREQUENCY_RESOLUTION < (17.32*pow(2,p))+ pow(2,p-1)*1.03*sensitivity_coefficient):
            return 1 #Note = C#

        elif (18.35*pow(2,p)-pow(2,p-1)*1.10*sensitivity_coefficient < ind*FREQUENCY_RESOLUTION and ind*FREQUENCY_RESOLUTION < 18.35*pow(2,p)+pow(2,p-1)*1.10*sensitivity_coefficient):
            return 2 #Note = D

        elif (19.45*pow(2,p)- pow(2,p-1)*1.15*sensitivity_coefficient < ind*FREQUENCY_RESOLUTION and ind*FREQUENCY_RESOLUTION < (19.45*pow(2,p))+pow(2,p-1)*1.15*sensitivity_coefficient):
             return 3 #Note = D#

        elif (20.60*pow(2,p)- pow(2,p-1)*1.23*E*sensitivity_coefficient < ind*FREQUENCY_RESOLUTION and ind*FREQUENCY_RESOLUTION < (20.60*pow(2,p))+pow(2,p-1)*1.23*E*sensitivity_coefficient):
            return 4; #Note = E

        elif (21.83*pow(2,p)- pow(2,p-1)*1.29*sensitivity_coefficient < ind*FREQUENCY_RESOLUTION and ind*FREQUENCY_RESOLUTION < (21.83*pow(2,p))+pow(2,p-1)*1.29*sensitivity_coefficient):
            return 5; #Note = F

        elif (23.12*pow(2,p)- pow(2,p-1)*1.38*sensitivity_coefficient < ind*FREQUENCY_RESOLUTION and ind*FREQUENCY_RESOLUTION < (23.12*pow(2,p))+pow(2,p-1)*1.38*sensitivity_coefficient):
            return 6; #Note = F#

        elif (24.50*pow(2,p)- pow(2,p-1)*1.46*sensitivity_coefficient < ind*FREQUENCY_RESOLUTION and ind*FREQUENCY_RESOLUTION < (24.50*pow(2,p))+pow(2,p-1)*1.46*sensitivity_coefficient):
            return 7; #Note = G

        elif (25.96*pow(2,p)- pow(2,p-1)*1.54*sensitivity_coefficient < ind*FREQUENCY_RESOLUTION and ind*FREQUENCY_RESOLUTION < (25.96*pow(2,p))+pow(2,p-1)*1.54*sensitivity_coefficient):
            return 8; #Note = G#

        elif (27.50*pow(2,p)- pow(2,p-1)*1.64*E*sensitivity_coefficient < ind*FREQUENCY_RESOLUTION and ind*FREQUENCY_RESOLUTION < (27.50*pow(2,p))+pow(2,p-1)*1.64*E*sensitivity_coefficient):
            return 9; #Note = A

        elif (29.14*pow(2,p)- pow(2,p-1)*1.73*sensitivity_coefficient < ind*FREQUENCY_RESOLUTION and ind*FREQUENCY_RESOLUTION < (29.14*pow(2,p))+pow(2,p-1)*1.73*sensitivity_coefficient):
            return 10; #Note = A#

        elif (30.87*pow(2,p)- pow(2,p-1)*1.83*E*sensitivity_coefficient < ind*FREQUENCY_RESOLUTION and ind*FREQUENCY_RESOLUTION < (30.87*pow(2,p))+pow(2,p-1)*1.83*E*sensitivity_coefficient):
            return 11; #Note = B

def plot_frequency_spectrum(originalWAV, alteredWAV,xRange):
    x = np.take(originalWAV, [*range(xRange)])
    y = np.take(alteredWAV, [*range(xRange)])
    fig, ax = plt.subplots()
    ax.plot(x, label="Original WAV")
    ax.plot(y, label="Altered WAV")
    ax.set(xlabel='Frequency Index', ylabel='Ampitude',
    title='Frequency Spectrum of Original and Altered WAV File')
    ax.grid()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    fig.savefig("freq_spectrum.png")
    #plt.show()

def import_cache_file(filename):
    with open(filename) as infile:
        return json.load(infile)

def convert_key(conversion_dict):
    wav_data_out_left = []
    wav_data_out_right = []
    current_duration = 0
    target_notes = conversion_dict.keys()
    while current_duration + BUFFER_SIZE < left_audio_data.size:
        next_duration = current_duration + BUFFER_SIZE
        print("current_duration:", current_duration, "next_duration:", next_duration)
        left_audio_bin = left_audio_data[current_duration:next_duration]
        right_audio_bin = right_audio_data[current_duration:next_duration]
        current_duration = next_duration

        # Compute the FFT to get the frequency-domain representation
        frequency_spectrum_left = np.fft.fft(left_audio_bin)
        frequency_spectrum_right = np.fft.fft(right_audio_bin)
        print("frequency_spectrum_left.size:", frequency_spectrum_left.size)

        # Shift frequencies of C major notes to those of C minor
        for f in range(int(BUFFER_SIZE / 2)):
            #f denotes the index of the fft array
            current_pitch = pitchRecognize(f)
            current_freq = f*FREQUENCY_RESOLUTION
            if current_pitch in target_notes:
                #Specify the pitch that is a half-step down from the current pitch
                #The FFT is symmetric so we have to handle both halves, symmetrically
                symmetric_f = BUFFER_SIZE - f
                #Check whether the note in question needs to be raised or lowered,
                #based on which scale mode conversion is being made
                if conversion_dict[current_pitch] == 0:
                    new_key_f = int(f/HALF_STEP_RATIO)
                else:
                    new_key_f = int(f*HALF_STEP_RATIO)

                symmetric_new_key_f = BUFFER_SIZE - new_key_f
                temp_left = frequency_spectrum_left[new_key_f]
                temp_right = frequency_spectrum_right[new_key_f]
                frequency_spectrum_left[new_key_f] = frequency_spectrum_left[f]
                frequency_spectrum_right[new_key_f] = frequency_spectrum_right[f]
                frequency_spectrum_left[symmetric_new_key_f] = frequency_spectrum_left[symmetric_f]
                frequency_spectrum_right[symmetric_new_key_f] = frequency_spectrum_right[symmetric_f]
                frequency_spectrum_left[f] = temp_left
                frequency_spectrum_left[symmetric_f] = temp_left
                frequency_spectrum_right[f] = temp_right
                frequency_spectrum_right[symmetric_f] = temp_right

        # Take the inverse FFT to get the modified audio back, and append it to running wav data list
        modified_wav_clip_left = np.real(np.fft.ifft(frequency_spectrum_left))
        modified_wav_clip_left = np.float32(modified_wav_clip_left).tolist()
        wav_data_out_left = wav_data_out_left + modified_wav_clip_left
        modified_wav_clip_right = np.real(np.fft.ifft(frequency_spectrum_right))
        modified_wav_clip_right = np.float32(modified_wav_clip_right).tolist()
        wav_data_out_right = wav_data_out_right + modified_wav_clip_right

    wav_data_out = [[wav_data_out_left[i], wav_data_out_right[i]] for i in [*range(len(wav_data_out_left))]]
    return wav_data_out

if __name__ == '__main__':
    main()
