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
directory$ = "'directory$'" + "/"

file_pattern$ = directory$*file_type$

Create Strings as file list: "fileList", file_pattern$

numberOfFiles = Get number of strings

#gets the file name for later use
for i from 1 to numberOfFiles
    selectObject: "Strings fileList"

    fileName$ = Get string: i

    baseName$ = fileName$ - ".'file_type$'"

	select Strings list
        filename$ = Get string... 'i'
        Read from file... 'directory$''filename$'
        soundname$ = selected$ ("Sound")

	filedur = Get total duration
	# identify associated TextGrid
	gridfile$ = "'directory$''soundname$'.TextGrid"

	if fileReadable (gridfile$)
		Read from file... 'gridfile$'
		select TextGrid 'soundname$'
		number_intervals = Get number of intervals... silence_tier

		start$ = ""
		end$ = ""
		for k from 1 to number_intervals
			select TextGrid 'soundname$'
			end$ = Get label of interval... silence_tier 'k'

			if end$ == "sil" or end$ == "{sil}"
			    if (start$ <> "sil"or start$ <> "{sil}") and start$ <> ""
				    start_time = Get start time of interval: silence_tier, start$
				else
				    start_time = Get end time of interval: silence_tier, start$
				endif

				end_time = Get start time of interval: silence_tier, end$

				selectObject: 'soundname$'
				chunk$ = Extract part: start_time, end_time, "rectangular", 1, "no"
				start$ = end$
				Save as WAV file: chunk$, 'directory$'+'start_time'+'file_type$'
			endif
		endfor
	endif
    appendInfoLine: "Processing file: ", fileName$
endfor
