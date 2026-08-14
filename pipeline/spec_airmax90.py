# -*- coding: utf-8 -*-
"""The Air Max 90 lasted-upper inspection spec, as data.

Transcribed from the 2019 Smart Vision Tech / SHC Global review deck. Each entry
keeps the vendor's own sensing plan (which camera, 2D or 3D) alongside what our
demo can honestly compute from the photographs we actually have, so the UI can
show an item as measured, degraded, or simply not sensable rather than quietly
skipping it.

feasibility:
  measured   computed from the view we have, against a golden reference
  needs_view we implement the algorithm but lack that camera angle in the data
  needs_3d   geometrically impossible from a single 2D image
  advisory   a material/colour capability judgement, not a per-unit measurement
"""

CAMERAS = {
    "CAM1": {"name": "전면부 2D", "role": "toe front"},
    "CAM2": {"name": "측면부 3D", "role": "toe profile (3D)"},
    "CAM3": {"name": "상부 2D", "role": "top / eyestay"},
    "CAM4": {"name": "후면부 2D", "role": "rear / heel"},
    "CAM5": {"name": "하부 2D", "role": "bottom / strobel"},
    "CAM6": {"name": "내측면부 2D", "role": "medial side"},
    "CAM7": {"name": "외측면부 2D", "role": "lateral side"},
}

# view each item is measurable from, in our data
LATERAL, TOP, REAR, BOTTOM = "lateral", "top", "rear", "bottom"

ITEMS = [
    {
        "id": "tip_length",
        "no": 1,
        "name_en": "Tip Length",
        "name_ko": "팁 길이",
        "sensing": "2D+3D",
        "cameras": ["CAM1", "CAM2"],
        "view": LATERAL,
        "feasibility": "needs_3d",
        "part": "toe",
        "vendor_logic": (
            "팁 라인에 2차방정식을 피팅해 변곡점을 추출하고, 밑창 상단 평행라인과 "
            "팁 타원의 교점을 구한 뒤, 변곡점~팁 상부의 가상 경로에 굴곡 가중치를 "
            "부여해 실제 굴곡 거리로 환산. 정면에서 가려지는 구간은 3D 측면 형상과 "
            "중합해 실거리로 보정."
        ),
        "our_method": (
            "토 윤곽에 2차곡선을 피팅해 변곡점을 찾고 밑창선과의 교점까지 "
            "곡률 가중 호장(arc length)을 적분한다. 단일 2D 사진에서는 전방 "
            "단축(foreshortening)을 풀 수 없어 참값이 아닌 **상대 지표**로만 제시."
        ),
        "unit": "px",
    },
    {
        "id": "tip_center",
        "no": 2,
        "name_en": "Tip Center",
        "name_ko": "팁 센터",
        "sensing": "2D",
        "cameras": ["CAM1", "CAM2"],
        "view": LATERAL,
        "feasibility": "measured",
        "part": "toe",
        "vendor_logic": "팁 중심(변곡점)부터 스트랩 하단부까지의 거리 연산.",
        "our_method": (
            "토 윤곽의 곡률 최대점을 팁 중심으로 잡고, 아일릿/레이스 영역 "
            "하단선까지의 거리를 잰다. 골든 샘플 대비 편차로 판정."
        ),
        "unit": "px",
    },
    {
        "id": "eyestay_width",
        "no": 3,
        "name_en": "Eyestay Opening Width",
        "name_ko": "아이스테이 개구폭",
        "sensing": "2D",
        "cameras": ["CAM3"],
        "view": TOP,
        "feasibility": "needs_view",
        "part": "eyestay",
        "vendor_logic": "지정 색상(레이스 사이로 보이는 안감)을 추출한 뒤 좌우간 거리 계측.",
        "our_method": (
            "상부 뷰에서 레이스 사이 개구 영역을 색·명도로 분리하고, 각 개구의 "
            "좌우 극점 사이 폭을 3구간에서 계측해 좌우 대칭성까지 본다."
        ),
        "unit": "px",
    },
    {
        "id": "forefoot_mudguard",
        "no": 4,
        "name_en": "Forefoot Mudguard Distance",
        "name_ko": "전족부 머드가드 거리",
        "sensing": "2D",
        "cameras": ["CAM6", "CAM7"],
        "view": LATERAL,
        "feasibility": "measured",
        "part": "mudguard",
        "vendor_logic": "머드가드 라인과 바닥라인을 각각 직선 검출해 두 직선 사이 거리를 산출.",
        "our_method": (
            "전족부 구간에서 바닥선과 머드가드 상단 경계를 선분 검출로 뽑아 "
            "평행 근사한 뒤 수직 거리를 측정."
        ),
        "unit": "px",
    },
    {
        "id": "heel_mudguard",
        "no": 5,
        "name_en": "Heel Mudguard Distance",
        "name_ko": "힐 머드가드 거리",
        "sensing": "2D",
        "cameras": ["CAM6", "CAM7"],
        "view": LATERAL,
        "feasibility": "measured",
        "part": "mudguard",
        "vendor_logic": "바닥 라인 추출 후 Heel Mudguard 라인 검출, 두 라인의 높이 차.",
        "our_method": "힐 구간에서 항목 4와 같은 방식으로 바닥선~머드가드선 거리를 측정.",
        "unit": "px",
    },
    {
        "id": "heel_height",
        "no": 6,
        "name_en": "Heel Height",
        "name_ko": "힐 높이",
        "sensing": "2D",
        "cameras": ["CAM4"],
        "view": REAR,
        "feasibility": "measured",
        "part": "heel",
        "vendor_logic": "힐 외벽라인의 코너부를 감지해 교점 또는 코너검출 알고리즘으로 높이 산출.",
        "our_method": (
            "힐 외곽 윤곽에서 곡률 극대점(카운터 상단 코너)을 찾아 바닥선까지의 "
            "수직 거리를 측정. 후면 뷰가 있으면 그것을, 없으면 측면 뷰로 대체하고 "
            "대체 사실을 표시."
        ),
        "unit": "px",
    },
    {
        "id": "heel_overlay",
        "no": 7,
        "name_en": "Heel Overlay Distance",
        "name_ko": "힐 오버레이 거리",
        "sensing": "2D",
        "cameras": ["CAM4"],
        "view": REAR,
        "feasibility": "measured",
        "part": "heel",
        "vendor_logic": "힐 부분 머드가드를 사선 검출로 잡고 두 사선의 교점을 채용(원본 사전 변조 필요).",
        "our_method": (
            "힐 영역에서 두 개의 우세 사선을 선분 검출로 추출해 교점을 구하고 "
            "바닥선까지의 거리를 측정."
        ),
        "unit": "px",
    },
    {
        "id": "heel_center",
        "no": 8,
        "name_en": "Heel Center",
        "name_ko": "힐 센터",
        "sensing": "2D",
        "cameras": ["CAM4"],
        "view": REAR,
        "feasibility": "needs_view",
        "part": "heel",
        "vendor_logic": "후면 뷰에서 힐 중심 정렬 확인.",
        "our_method": (
            "후면 뷰 실루엣의 좌우 대칭축과 힐 카운터 중앙선(로고/솔기 기준)의 "
            "수평 편차를 측정. 측면 뷰로는 정의되지 않는 항목."
        ),
        "unit": "px",
    },
    {
        "id": "strobel",
        "no": 9,
        "name_en": "Strobel Inspection",
        "name_ko": "중창(스트로벨) 검사",
        "sensing": "2D",
        "cameras": ["CAM5"],
        "view": BOTTOM,
        "feasibility": "needs_view",
        "part": "strobel",
        "vendor_logic": (
            "윤곽 추출·증폭 → 고대비 재처리 → 잡음 제거/분리. Strobel 중심과 중심선, "
            "좌우 종단을 잡아 종단폭 대비 4분할 위치를 정하고 3포인트 폭을 계측. "
            "추가로 홀 내부 RED MARK를 색상 선택 추출 + 최소크기 제어로 감지."
        ),
        "our_method": (
            "하부 뷰에서 스트로벨 외곽을 분리해 중심선을 잡고 4분할 3지점 폭을 "
            "계측, 홀은 최소면적 제어 블롭 검출로 찾고 홀 내부 색을 판정."
        ),
        "unit": "px",
    },
    {
        "id": "colorway_capability",
        "no": 10,
        "name_en": "Other Colourway Capability",
        "name_ko": "타색상 검사 가능성",
        "sensing": "-",
        "cameras": [],
        "view": LATERAL,
        "feasibility": "advisory",
        "part": None,
        "vendor_logic": (
            "색상보다 재질의 영향이 크다. 광택 소재와 엠보싱 패턴은 검사 수준이 "
            "대폭 저하될 가능성이 있음."
        ),
        "our_method": (
            "정반사(스펙큘러) 화소 비율과 텍스처 에너지를 측정해 광택·엠보싱 정도를 "
            "정량화하고, 그 값으로 해당 개체의 검사 난이도를 등급화."
        ),
        "unit": "%",
    },
    {
        "id": "black_capability",
        "no": 11,
        "name_en": "Black Colourway Capability",
        "name_ko": "검정색 검사 가능성",
        "sensing": "-",
        "cameras": [],
        "view": LATERAL,
        "feasibility": "advisory",
        "part": None,
        "vendor_logic": "검사 가능하나 타색상 대비 약 90% 수준의 효율로 예측됨.",
        "our_method": (
            "어두운 영역의 국소 대비(에지 응답)와 계조 여유를 측정해 검정 소재에서 "
            "실제로 남는 검사 여력을 수치화."
        ),
        "unit": "%",
    },
]

BY_ID = {it["id"]: it for it in ITEMS}

RIG = {
    "cameras_2d": 6,
    "cameras_3d": 1,
    "cycle_sec": 8,
    "move_sec": 2,
    "inspect_sec": 2,
    "notes": [
        "제품을 상하 반전시킨 뒤 검사",
        "좌/우 한 Set 동시 검사, 좌/우 구분 없이 공급하고 앞/뒤만 구분",
        "사이즈 Grading에 따라 카메라 위치 조정",
        "LAST GUIDE PIN에 제품 장착, 더블 체인 2열로 요동 방지",
        "검사 신뢰를 위한 암실 구조",
    ],
}
