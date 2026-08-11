// Initialize jsPsych
const jsPsych = initJsPsych();

// Define a simple trial
const hello_trial = {
  type: jsPsychHtmlKeyboardResponse,
  stimulus: "<p>Hello! Press any key to finish.</p>",
};

// Run the experiment
jsPsych.run([hello_trial]);
