import json
import sys
import traceback

from vlm_planner import QwenVLPlanner


def main():
    try:
        payload = json.load(sys.stdin)
        planner = QwenVLPlanner()
        result = planner.plan(
            image_path=payload['image_path'],
            C_I=payload.get('C_I'),
            D_I=payload.get('D_I'),
            A_I=payload.get('A_I'),
            I=payload.get('I', payload.get('image_path')),
            prompt=payload.get('prompt'),
            perception_info=payload.get('perception_info', payload.get('clip_info', {})),
            direct_plan=payload.get('direct_plan', []),
            allowed_steps=payload.get('allowed_steps', []),
        )
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as exc:
        error = {
            'error': str(exc),
            'traceback': traceback.format_exc(limit=8),
        }
        print(json.dumps(error, ensure_ascii=False))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
