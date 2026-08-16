import pandas as pd
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .models import ExperimentResult
from .utils import audio


@csrf_exempt
def save_experiment_data(request):
    if request.method == "POST":
        try:
            data = pd.read_json(request.body).to_dict(orient='records')
            ExperimentResult.objects.create(data=data)
            return JsonResponse({"status": "success"}, status=200)
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=400)
    return JsonResponse({"status": "method not allowed"}, status=405)
