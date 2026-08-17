from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .models import audio_experiment_record
from .utils import transcription


@csrf_exempt
def handle_transcription(request):
    if request.method.lower() != "post":
        return JsonResponse({"ERROR": "METHOD NOT ALLOWED"}, status=405)

    uploaded_file = request.FILES.get("audio")
    if uploaded_file is None:
        return JsonResponse({"ERROR": "NO AUDIO FILE ATTACHED"}, status=400)

    subject = request.POST.get("subject", "")
    trial_index = request.POST.get("trial_index", "")
    stimulus = request.POST.get("stimulus", "")
    custom_tag = request.POST.get("custom_tag", "clear_speech")

    try:
        service = transcription()
        result = service.execute(uploaded_file)
        confidence_score = service.confidence

        record = audio_experiment_record.objects.create(
            subject=subject,
            trial_index=str(trial_index),
            stimulus=stimulus,
            response=result,
            custom_tag=custom_tag
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
