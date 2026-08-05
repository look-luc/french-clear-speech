form Extracting individual utterance
	comment Specify which tier the main tier where SIL is located:
		integer silence_tier 2
	comment Sound file extension:
		optionmenu file_type: 1
		option .wav
		option .mp3
		option .aiff
endform

directory$ = chooseDirectory$ ("Choose the directory containing sound files and textgrids")
directory$ = directory$ + "/"

output_dir$ = chooseDirectory$ ("Choose the directory where you want to save the files")

file_pattern$ = directory$ + "*" + file_type$

fileListObj = Create Strings as file list: "fileList", file_pattern$

selectObject: fileListObj
numberOfFiles = Get number of strings

#gets the file name for later use
for i from 1 to numberOfFiles
    selectObject: fileListObj
    fileName$ = Get string: i

    baseName$ = fileName$ - file_type$

    Read from file: directory$ + fileName$
    soundname$ = selected$ ("Sound")
    resultfile$ = directory$ + soundname$ + ".txt"

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
                if start_sil$ <> "sil" and start_sil$ <> "{sil}" and start_sil$ <> ""
                    start_time = Get start time of interval: silence_tier, k
                else
                    start_time = Get start time of interval: silence_tier, k
                endif

                end_time = Get end time of interval: silence_tier, k

                selectObject: "Sound " + soundname$
                chunkObj = Extract part: start_time, end_time, "rectangular", 1, "no"

                selectObject: "TextGrid " + soundname$
                startIndex = Get interval at time: silence_tier, start_time
                endIndex = Get interval at time: silence_tier, end_time

                sweep$ = ""
                for j from startIndex to endIndex
                    selectObject: "TextGrid " + soundname$
                    interval_label$ = Get label of interval: silence_tier, j
                    sweep$ = sweep$ + interval_label$ + " "
                endfor

                fileappend "'resultfile$'" 'sweep$'
                start_sil$ = end_sil$

                out_path$ = output_dir$ + baseName$ + "_" + string$(k) + ".wav"
                selectObject: chunkObj
                Save as WAV file: out_path$

                removeObject: chunkObj
            endif
        endfor
        removeObject: "TextGrid " + soundname$
    endif
    removeObject: "Sound " + soundname$
    appendInfoLine: "Processing file: ", fileName$
endfor

removeObject: fileListObj
