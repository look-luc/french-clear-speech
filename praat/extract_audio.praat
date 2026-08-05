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

file_pattern$ = directory$ + "*" + file_type$

fileListObj = Create Strings as file list: "fileList", file_pattern$

selectObject: fileListObj
numberOfFiles = Get number of strings

# gets the file name for later use
for i from 1 to numberOfFiles
    selectObject: fileListObj
    fileName$ = Get string: i

    baseName$ = fileName$ - file_type$

    Read from file: directory$ + fileName$
    soundname$ = selected$ ("Sound")

    filedur = Get total duration
    # identify associated TextGrid
    gridfile$ = directory$ + soundname$ + ".TextGrid"

    if fileReadable (gridfile$)
        Read from file: gridfile$
        selectObject: "TextGrid " + soundname$
        number_intervals = Get number of intervals: silence_tier

        start_sil$ = ""
        end_sil$ = ""
        for k from 1 to number_intervals
            selectObject: "TextGrid " + soundname$
            end_sil$ = Get label of interval: silence_tier, k

            if end_sil$ == "sil" or end_sil$ == "{sil}"
                if k > 1
                    start_time = Get end time of interval: silence_tier, k - 1
                else
                    start_time = 0
                endif

                end_time = Get start time of interval: silence_tier, k

                if (end_time - start_time) > 0.005
                    selectObject: "Sound " + soundname$
                    chunkObj = Extract part: start_time, end_time, "rectangular", 1, "no"

                    out_path$ = output_dir$ + baseName$ + "_" + string$(k) + ".wav"
                    selectObject: chunkObj
                    Save as WAV file: out_path$

                    removeObject: chunkObj
                endif

                start_sil$ = end_sil$
            endif
        endfor
        removeObject: "TextGrid " + soundname$
    endif
    removeObject: "Sound " + soundname$
    appendInfoLine: "Processing file: ", fileName$
endfor

removeObject: fileListObj
