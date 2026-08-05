form Extracting individual utterance
	comment Specify which tier the main tier where SIL is located:
		integer silence_tier 2
	comment Specify which tier is the one with the transcription:
	    integer transcription_tier 2
	comment Sound file extension:
		optionmenu file_type: 1
		option .wav
		option .mp3
		option .aiff
endform

directory$ = chooseDirectory$ ("Choose the directory containing sound files and textgrids")
directory$ = directory$ + "/"

# Subfolder path inside directory$
output_dir$ = directory$ + "data_sounds/"
createDirectory: output_dir$

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
    resultfile$ = output_dir$ + soundname$ + ".txt"

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
                    if start_sil$ <> "sil" and start_sil$ <> "{sil}" and start_sil$ <> ""
                        start_time = Get end time of interval: silence_tier, k - 1
                    else
                        start_time = Get end time of interval: silence_tier, k - 1
                    endif
                else
                    start_time = 0
                endif

                end_time = Get start time of interval: silence_tier, k
                if end_time > start_time
                    selectObject: "Sound " + soundname$
                    chunkObj = Extract part: start_time, end_time, "rectangular", 1, "no"

                    # Fetch interval boundaries with a 1ms offset to avoid exact line collisions
                    selectObject: "TextGrid " + soundname$
                    startIndex = Get interval at time: transcription_tier, start_time + 0.001
                    endIndex = Get interval at time: transcription_tier, end_time - 0.001

                    sweep$ = ""
                    for j from startIndex to endIndex
                        selectObject: "TextGrid " + soundname$
                        interval_label$ = Get label of interval: transcription_tier, j
                        sweep$ = sweep$ + interval_label$ + " "
                    endfor

                    appendFile: resultfile$, sweep$, newline$
                    start_sil$ = end_sil$

                    out_path$ = output_dir$ + baseName$ + "_" + string$(k) + ".wav"
                    selectObject: chunkObj
                    Save as WAV file: out_path$

                    removeObject: chunkObj
                endif
            endif
        endfor
        removeObject: "TextGrid " + soundname$
    endif
    removeObject: "Sound " + soundname$
    appendInfoLine: "Processing file: ", fileName$
endfor

removeObject: fileListObj
