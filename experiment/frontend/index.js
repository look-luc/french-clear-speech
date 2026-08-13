const consentGranted = sessionStorage.getItem("consent_granted");
if (consentGranted !== "true") {
  window.location.href = "consent.html";
} else {
  var jsPsych = initJsPsych({
    on_finish: function () {
      sessionStorage.removeItem("consent_granted");
      jsPsych.data.displayData();
    },
  });

  jsPsych.data.addProperties({
    consent_given: true,
    consent_timestamp: sessionStorage.getItem("consent_timestamp"),
  });
  var timeline = [];

  var init_mic = {
    type: jsPsychInitializeMicrophone,
    device_select_message: "Please select your microphone:",
    button_label: "Use this microphone",
  };
  timeline.push(init_mic);

  var timerInterval;

  var test_mic = {
    type: jsPsychHtmlAudioResponse,
    stimulus: `
        <div class="recording-container">
          <span class="rec-dot"></span>
          <span class="rec-text">Recording: </span>
          <span id="timer-display" class="timer">5 seconds</span>
        </div>
        <p>Please say <strong>"test"</strong> to check your microphone.</p>
      `,
    recording_duration: 5000,
    allow_playback: true,
    on_load: function () {
      var secondsLeft = 5;
      timerInterval = setInterval(function () {
        secondsLeft -= 1;
        var timerDisplay = document.getElementById("timer-display");
        if (timerDisplay && secondsLeft >= 0) {
          timerDisplay.innerText = secondsLeft + " seconds";
        } else {
          clearInterval(timerInterval);
        }
      }, 1000);
    },
    on_finish: function () {
      clearInterval(timerInterval);
    },
  };
  timeline.push(test_mic);

  var welcome = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: "Welcome to the experiment. Press any key to begin.",
  };
  timeline.push(welcome);

  jsPsych.run(timeline);
}
