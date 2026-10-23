from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .models import Audio, DataResponse, DataResponseAudio
from .utils import transcription


@csrf_exempt
def handle_transcription(request):
    if request.method.lower() != "post":
        return JsonResponse({"ERROR": "METHOD NOT ALLOWED"}, status=405)

    uploaded_file = request.FILES.get("audio")
    snr_type = request.FILES.get("situation")
    if uploaded_file is None:
        return JsonResponse({"ERROR": "NO AUDIO FILE ATTACHED"}, status=400)

    subject = request.POST.get("subject", "")
    age_raw = request.POST.get("age", "")
    age = int(age_raw) if age_raw.isdigit() else 0
    gender = request.POST.get("gender", "")
    trial_index = request.POST.get("trial_index", "")
    stimulus = request.POST.get("stimulus", "")
    custom_tag = request.POST.get("custom_tag", "clear_speech")

    try:
        service = transcription()
        result, confidence_score = service.execute(
            file=uploaded_file,
            noise_type=snr_type
        )

        audio_instance = Audio.objects.create(
            audio_str=uploaded_file.name
        )

        record = DataResponse.objects.create(
            subject = subject,
            age = age,
            gender = gender,
            trial_index = trial_index,
            response = stimulus,
            confidence = confidence_score,
            custom_tag = custom_tag
        )

        DataResponseAudio.objects.create(
            data_response = record,
            audio = audio_instance
        )

        return JsonResponse(
            {
                "STATUS": "SUCCESS",
                "PRIMARY_KEY": record.primary_key,
                "TRANSCRIPTION": result,
                "CONFIDENCE": confidence_score
            },
            status=200
        )
    except Exception as e:
        return JsonResponse({"STATUS": "ERROR", "MESSAGE": str(e)}, status=500)
