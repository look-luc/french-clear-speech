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

    gridfile$ = directory$ + soundname$ + ".TextGrid"

    if fileReadable (gridfile$)
        Read from file: gridfile$
        selectObject: "TextGrid " + soundname$
        number_intervals = Get number of intervals: silence_tier

        has_seen_first_sil = 0
        prev_sil_end = 0.0
        utterance_index = 1

        for k from 1 to number_intervals
            selectObject: "TextGrid " + soundname$
            label$ = Get label of interval: silence_tier, k

            if label$ == "SIL" or label$ == "{sl}"
                sil_start = Get start time of interval: silence_tier, k
                sil_start_int = Get label of interval: silence_tier, k

                if has_seen_first_sil and sil_start > prev_sil_end
                    selectObject: "Sound " + soundname$
                    partial_sound = Extract part: prev_sil_end, sil_start, "rectangular", 1.0, "no"

                    file_name$ = "DATA_" + soundname$ + "_" + string$(utterance_index) + ".wav"
                    Save as WAV file: output_dir$ + file_name$

                    resultfile$ = output_dir$+ "DATA_" + soundname$ + "_" + string$(utterance_index) +  ".txt"
                    for j from prev_sil_int to sil_start_int
                        appendFile: resultfile$,  j + " "
                    endfor

                    removeObject: partial_sound
                    utterance_index = utterance_index + 1
                endif

                prev_sil_end = Get end time of interval: silence_tier, k
                prev_sil_int = Get label of interval: silence_tier, k
                has_seen_first_sil = 1
            endif
        endfor

        removeObject: "TextGrid " + soundname$
    endif

    removeObject: "Sound " + soundname$
    appendInfoLine: "Processing file: ", filename$
endfor

removeObject: fileListObj
