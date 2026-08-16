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

  var subject_id = jsPsych.randomization.randomID(10);

  jsPsych.data.addProperties({
    subject: subject_id,
  });

  jsPsych.data.addProperties({
    consent_given: true,
    consent_timestamp: sessionStorage.getItem("consent_timestamp"),
  });
  var timeline = [];

  var init_mic = {
    type: jsPsychInitializeMicrophone,
    device_select_message:
      "Please select your microphone/S'il vous plaît choisir votre microphone:",
    button_label: "Use this microphone/Utilisez cette microphone",
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
        <p>S'il vous plaît dis <strong>"test"</strong> pour le vérification votre microphone
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
    stimulus: `<div class="welcome-eng">
        <p><strong>Welcome to the experiment.</strong></p>
        <p>You will be prompted to say a phrase as clear as possible given the situation that the phrase is in. You have only ONE (1) chance to record. Once you are done recording, a transcription will show with the percentage of what the "listener" heard what you said.</p>
        <p>Press any key to begin.</p>
        <p><strong>Bienvenue a l'expérience.</strong></p>
        <p>Vous serez invité à dire une phrase aussi clair que possible pendant la situation que la phrase est dans. Vous avez juste UNE (1) chance pour enregistrer. Quand vous avez terminé, une transcription va montrer avec le pourcentage de quoi la "personne qui l'écoute" a écouté que vous avez dire.</p>
        <p>Poussez n'importe quelle touche pour commencer.</p>
      </div>`,
  };
  timeline.push(welcome);

  var test_run = {
    type: jsPsychHtmlAudioResponse,
    stimulus: jsPsych.timelineVariable("stimulus"),
    recording_duration: null,
    show_done_button: true,
    done_button_label: "Stop Recording/Arretez Enregistrer",
    data: {
      custom_tag: "clear-speech",
    },
    on_finish: async function (data) {
      audio_blob = data.audio_response;
      (transcription_text,
        (confidence = await upload_and_transcribe(audio_blob)));
      data.model_transcript = transcription_text;
      data.confidence = confidence;
    },
  };

  var jspsych_display_trial = {
    type: jsPsychHtmlButtonResponse,
    stimulus: function () {
      prev_data = jsPsych.get().last(1).values()[0];
      text = prev_data.model_transcript;
      confidence = data.confidence * 100;
      return (
        "<div>Je suis <b>" +
        confidence +
        "</b>% sûr que vous avez dit: " +
        text +
        "</b></div>"
      );
    },
    choises: ["continue"],
  };

  const test_procedure = {
    timeline: [test_run],
    timeline_variables: STIMULI,
    randomize_order: true,
  };

  timeline.push(test_procedure);

  jsPsych.run(timeline);
}
