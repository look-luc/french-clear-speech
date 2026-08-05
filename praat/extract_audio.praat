form Extracting individual utterance
	comment Specify which tier the main tier where SIL is located:
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

Create Strings as file list... list 'directory$'*'file_type$'

number_files = Get number of strings

# gets the file name for later use
for i from 1 to number_files
    select Strings list
        filename$ = Get string... 'j'
        Read from file... 'directory$''filename$'
        soundname$ = selected$ ("Sound")
	filedur = Get total duration
	# identify associated TextGrid
	gridfile$ = "'directory$''soundname$'.TextGrid"

    # identify associated TextGrid
    gridfile$ = directory$ + soundname$ + ".TextGrid"

    if fileReadable (gridfile$)
        Read from file... 'gridfile$'
		select TextGrid 'soundname$'
		number_intervals = Get number of intervals... silence_tier

        prev_time = 0.0
        utterance_index = 1
        for k from 1 to number_intervals
            selectObject: "TextGrid " + soundname$
            lanel$ = Get label of interval: silence_tier, k

            if label$ == "SIL" or label$ == "{sl}"
                curr_Time = Get start point: silence_tier, k

                if curr_time > prev_time
                    select Sound 'soundname$'+ file_type$

                    partial_sound = Extract part: prev_time, curr_time, "rectangular", 1.0, "no"

                    file_name = "DATA_" + soundname$ + "_" + string$ (utterance_index) + ".wav"

                    Save as WAV file: output_dir$ + file_name
                    removeObject partial_sound
                    utterance_index = utterance_index + 1
                endif
                prev_time = curr_Time
            endif
        endfor
        if prev_time < filedur
            select Sound soundname$ + file_type$
            partial_sound = Extract part: prev_time, filedur, "rectangular", 1.0, "no"

            file_name = "DATA_" + soundname$ + "_" + string$ (utterance_index) + ".wav"

            Save as WAV file: output_dir$ + file_name
            removeObject partial_sound
        endif
        removeObject: "TextGrid " + soundname$
    endif
    removeObject: "Sound " + soundname$
    appendInfoLine: "Processing file: ", fileName$
endfor

removeObject: fileListObj
