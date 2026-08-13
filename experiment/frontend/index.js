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

  var test_mic = {
    type: jsPsychHtmlAudioResponse,
    stimulus: `
        <div class="recording-container">
          <span class="rec-dot"></span>
          <span class="rec-text">Recording <p id="record"></p></span>
        </div>
        <p>Please say <strong>"test"</strong> to check your microphone.</p>
      `,
    recording_duration: 5000,
    allow_playback: true,
  };

  var countDownDate = new Date();
  countDownDate.setSeconds(currentTime.getSeconds() + 5);
  var x = setInterval(function () {
    var now = new Date().getTime();

    var distance = countDownDate - now;

    var days = Math.floor(distance / (1000 * 60 * 60 * 24));
    var hours = Math.floor(
      (distance % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60),
    );
    var minutes = Math.floor((distance % (1000 * 60 * 60)) / (1000 * 60));
    var seconds = Math.floor((distance % (1000 * 60)) / 1000);

    document.getElementById("record").innerHTML =
      days + "d " + hours + "h " + minutes + "m " + seconds + "s ";

    if (distance < 0) {
      clearInterval(x);
      document.getElementById("record").innerHTML = "EXPIRED";
    }
  }, 1000);

  timeline.push(test_mic);

  var welcome = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: "Welcome to the experiment. Press any key to begin.",
  };
  timeline.push(welcome);

  jsPsych.run(timeline);
}
