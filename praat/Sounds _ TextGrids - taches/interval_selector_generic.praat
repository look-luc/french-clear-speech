########################################################################################
#  Interval Selector Script
########################################################################################
#  This script adds a "target" tier (tier 1) to mark the target interval in a long textgrid.
#  Each labeled interval on a given tier is compared against a list of targets, and if the
#  interval is on the target list, the interval is copied to the "target" tier, along with 
#  its label.
#
#  Input:   TextGrids with labeled tiers (presumably output from forced
#              aligner)
#  Output:  TextGrids with an added tier, named according to the original file.
#  
#  Additional files:   targets.txt (or similar) with a list of targets (e.g., words or phones)
#				 *must be in a "resources" folder inside the folder containing this script*
#
#  Process: The script asks for a directory in which to look for textgrid files, and a tier
#           number in which to search for targets. It then looks for TextGrids in the specified 
#           folder. For each textgrid, first, a new tier (tier 1) is created to mark targets.  
#	        Then, labeled intervals in the word tier are located. For labels that are listed
#           in the targets file, the script then copies that interval and label to the new 
#           target tier. Each new textgrid is saved as a new .textgrid file with the appendix  
#           "-target".  After all intervals in all files inthe specified directory have been 
#           examined, a finish message appears.
########################################################################################

form Marking target vowels
   comment Specify which tier in the original TextGrid is the search tier:
        integer word_tier 1
   comment Specify the name of the file (in the "resources" folder) listing target words:
		text targetfile targets.txt
endform

# File Chooser Dialog
directory$ = chooseDirectory$ ("Choose the directory containing textgrids")
directory$ = "'directory$'" + "/" 

# Specify output directory
out_dir$ = directory$

# Specify textgrid appendix
gridappendix$ = "-target"

# Specify textgrid extension
file_type$ = ".textgrid"

## Need targefile (in "resources" folder inside script folder; specified above)
Read Strings from raw text file... resources/'targetfile$'
Sort
To WordList
Rename... targets

# reassigns search tier numbers in anticipation of adding a new tier #1
new_word_tier = word_tier + 1

clearinfo
Create Strings as file list... list 'directory$'*'file_type$'
number_of_files = Get number of strings

# Starting from here, add everything that should be repeated for each textgrid file
for j from 1 to number_of_files
        select Strings list
        filename$ = Get string... 'j'
        Read from file... 'directory$''filename$'
	
	gridname$ = selected$ ("TextGrid")
        
        select TextGrid 'gridname$'
		Insert interval tier... 1 target
        number_of_intervals = Get number of intervals... 'new_word_tier'

                # Go through all word intervals in the file
                for k from 1 to number_of_intervals
	   		 		select TextGrid 'gridname$'
	   					label$ = Get label of interval... 'new_word_tier' 'k'
	   					if label$ <> ""

			# Check if the word interval label is on the list of target words
			select WordList targets
			wordcheck = Has word... 'label$'
			
			if wordcheck = 1
				select TextGrid 'gridname$'

				word_start = Get starting point... 'new_word_tier' 'k'
				word_end = Get end point... 'new_word_tier' 'k'

					Insert boundary... 1 'word_start'
					Insert boundary... 1 'word_end'
					target_interval = Get interval at time... 1 'word_end'-0.01
					Set interval text... 1 'target_interval' 'label$'
			endif
	            
				select TextGrid 'gridname$'
				out_filename$ = "'out_dir$'" + "'gridname$'" + "'gridappendix$'"
				Write to text file... 'out_filename$'.textgrid
	
			endif
		 endif
                endfor
                
		 select all
                minus Strings list
				minus WordList targets
                Remove
endfor

select all
Remove
print All textgrids been examined.

