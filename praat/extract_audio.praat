form Extracting individual utterance
	comment Specify the tier where SIL is located:
		integer silence_tier 2
	comment Sound file extension:
		optionmenu file_type: 1
		option .wav
		option .mp3
		option .aiff
	sentence Output_directory ./output/
endform

output_dir$ = output_directory$
if right$ (output_dir$, 1) <> "/"
    output_dir$ = output_dir$ + "/"
endif
createDirectory: output_dir$

directory$ = chooseDirectory$ ("Choose the directory containing sound files and textgrids")
directory$ = directory$ + "/"

fileListObj = Create Strings as file list: "list", directory$ + "*" + file_type$
number_files = Get number of strings

for i from 1 to number_files
    selectObject: fileListObj
    filename$ = Get string: i
    Read from file: directory$ + filename$
    soundname$ = selected$ ("Sound")
    filedur = Get total duration

    gridfile$ = directory$ + soundname$ + ".TextGrid"

    if fileReadable (gridfile$)
        Read from file: gridfile$
        selectObject: "TextGrid " + soundname$
        number_intervals = Get number of intervals: silence_tier

        prev_time = 0.0
        utterance_index = 1

        for k from 1 to number_intervals
            selectObject: "TextGrid " + soundname$
            label$ = Get label of interval: silence_tier, k

            if label$ == "SIL" or label$ == "{sl}"
                sil_start = Get start time of interval: silence_tier, k
                sil_end = Get end time of interval: silence_tier, k

                if sil_start > prev_time
                    selectObject: "Sound " + soundname$
                    partial_sound = Extract part: prev_time, sil_start, "rectangular", 1.0, "no"

                    file_name$ = "DATA_" + soundname$ + "_" + string$(utterance_index) + ".wav"
                    Save as WAV file: output_dir$ + file_name$
                    removeObject: partial_sound
                    utterance_index = utterance_index + 1
                endif
                prev_time = sil_end
            endif
        endfor

        if prev_time < filedur
            selectObject: soundname$ + file_type
            partial_sound = Extract part: prev_time, filedur, "rectangular", 1.0, "no"

            file_name$ = "DATA_" + soundname$ + "_" + string$(utterance_index) + ".wav"
            Save as WAV file: output_dir$ + file_name$
            removeObject: partial_sound
        endif

        removeObject: "TextGrid " + soundname$
    endif

    removeObject: soundname$ + file_type
    appendInfoLine: "Processing file: ", filename$
endfor

removeObject: fileListObj
