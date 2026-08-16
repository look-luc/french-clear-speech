import pandas as pd
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .utils import transcription


@csrf_exempt
def handle_transcription(request):
    if request.method.lower() != "post":
        return JsonResponse({"ERROR": "METHOD NOT ALLOWED"}, status=405)

    uploaded_file = request.FILES.get("audio")
    if uploaded_file is None:
        return JsonResponse({"ERROR": "NO AUDIO FILE ATTACHED"}, status=400)

    try:
        service = transcription()
        result = service.execute(uploaded_file)

        return JsonResponse({"STATUS": "SUCCESS", "TRANSCRIPTION": result}, status=200)
    except Exception as e:
        return JsonResponse({"STATUS": "ERROR", "MESSAGE": str(e)}, status=500)
