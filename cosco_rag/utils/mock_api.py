import random
import time
import uuid
from datetime import datetime, timedelta

def mock_cosco_api(endpoint: str, params: dict) -> dict:
    time.sleep(0.1)
    if endpoint == "query_space":
        available = random.random() > 0.2
        query_space_msg = {
            "available": available,
            "space_left": random.randint(1, 30) if available else 0,
            "voyage": "COSCO SHIPPING 012W",
            "etd": (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
        }
        return query_space_msg
    elif endpoint == "submit_booking":
        booking_no = f"COSU{uuid.uuid4().hex[:10].upper()}"
        so_no = f"SO{booking_no[-8:]}"
        submit_booking_msg = {"success": True, "booking_no": booking_no, "so_no": so_no}
        return submit_booking_msg
    elif endpoint == "get_so_status":
        endpoint_msg = {"status": "RELEASED", "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        return endpoint_msg
    elif endpoint == "get_vgm_deadline":
        deadline = datetime.now() + timedelta(hours=36)
        get_vgm_deadline_msg = {"deadline": deadline.isoformat()}
        return get_vgm_deadline_msg
    elif endpoint == "submit_bl_draft":
        submit_bl_draft_msg = {"task_id": f"BL_{params.get('booking_no')}", "message": "提单草稿已提交审核"}
        return submit_bl_draft_msg
    elif endpoint == "track_container":
        track_container_msg = {"events": ["Gate Out", "Loaded on Vessel", "Sailed"]}
        return track_container_msg
    return {"error": "unknown endpoint"}