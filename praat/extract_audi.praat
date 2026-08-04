form Extracting individual utterance
	comment Specify which tier the main tier where SIL is located:
		integer silence_tier 2
	comment Sound file extension:
		optionmenu file_type: 1
		option .wav
		option .mp4
		option .aiff
endform

directory$ = chooseDirectory$ ("Choose the directory containing sound files and textgrids")
directory$ = "'directory$'" + "/"