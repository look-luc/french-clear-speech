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
        <p>S'il vous plaît dis <strong>"test"</strong> pour le vérification votre microphone</p>
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

  async function upload_and_transcribe(audio_blob, metadata) {
    const form_data = new FormData();
    form_data.append("audio", audio_blob, "recording.wav");
    form_data.append("subject", metadata.subject);
    form_data.append("trial_index", metadata.trial_index);
    form_data.append("stimulus", metadata.stimulus);
    form_data.append("custom_tag", metadata.custom_tag);

    const response = await fetch("/handle_transcription/", {
      method: "POST",
      body: form_data,
    });
    const results = await response.json();

    if (results.STATUS === "SUCCESS") {
      return [results.TRANSCRIPTION, results.CONFIDENCE];
    } else {
      throw new Error(results.MESSAGE);
    }
  }

  var audio_blob = null;
  var metadata = null;

  var test_run = {
    type: jsPsychHtmlAudioResponse,
    stimulus: jsPsych.timelineVariable("stimulus"),
    show_done_button: true,
    recording_duration: 30000,
    allow_playback: false,
    done_button_label: "Stop Recording/Arretez Enregistrer",
    data: {
      custom_tag: "clear_speech",
    },
    on_finish: function (data) {
      audio_blob = data.audio_response || data.response;

      metadata = {
        subject: subject_id,
        trial_index: data.trial_index,
        stimulus: data.stimulus,
        custom_tag: data.custom_tag,
      };
    },
  };

  var processing_audio = {
    type: jsPsychHtmlKeyboardResponse,
    choices: "NO_KEYS",
    stimulus: "<div> Transcribing audio/Transcrit l'audio... </div>",
    on_load: async function () {
      if (!audio_blob) {
        console.error("Audio blob is not available yet.");
        jsPsych.finishTrial({ model_transcript: "NO_AUDIO", confidence: 0 });
        return;
      }
      const blob = await fetch(audio_blob).then((r) => r.blob());
      const [transcription_text, confidence_val] = await upload_and_transcribe(
        blob,
        metadata,
      );
      jsPsych.finishTrial({
        model_transcript: transcription_text,
        confidence: confidence_val,
      });
    },
  };

  var jspsych_display_trial = {
    type: jsPsychHtmlButtonResponse,
    stimulus: function () {
      const prev_data = jsPsych.data.get().last(1).values()[0];
      const text = prev_data.model_transcript;
      const confidence_prct = Math.round((prev_data.confidence || 0) * 100);
      return (
        "<div>Je suis <b>" +
        confidence_prct +
        "%</b> sûr que vous avez dit: <b>" +
        text +
        "</b></div>"
      );
    },
    choices: ["Continue"],
  };

  const test_procedure = {
    timeline: [test_run, processing_audio, jspsych_display_trial],
    timeline_variables: STIMULI,
    randomize_order: true,
  };

  timeline.push(test_procedure);

  jsPsych.run(timeline);
}
