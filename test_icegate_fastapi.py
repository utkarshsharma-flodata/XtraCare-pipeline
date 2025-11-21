from typing import Optional, List, Dict, Any
from fastapi import FastAPI
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator
from bs4 import BeautifulSoup
import requests

headers = {"User-Agent": "Mozilla/5.0"}
base_icegate_url = "https://www.old.icegate.gov.in"

app = FastAPI(
    title="ICEGATE Tariff & CCR Data API",
    description="API to fetch Indian Harmonized System of Nomenclature (HSN) tariff and compulsory compliance (CCR) data from - https://www.old.icegate.gov.in/Webappl/Trade-Guide-on-Imports ",
    version="1.0.0"
)

class HSNRequest(BaseModel):
    """
    Payload for requesting tariff and CCR data for a list of HSN codes.
    """
    hsn_codes: List[str] = Field(
        title="HSN Codes",
        description="List of HSN codes to query with respesctive Country Name",
        example=["10019910","21011130"],
    )

    country: str | None = Field(
    None,
    title="Country Name",
    description="Optional country name (defaults to CHINA)",
    example="CHINA"
    )

    assessable_value: Optional[int] = Field(
        100000,
        title="Assessable Value",
        description="Value on which customs duty is calculated (default: 100000)",
        example=100000
    )

    quantity: Optional[int] = Field(
        100,
        title="Quantity",
        description="Quantity of goods (default: 100)",
        example=100
    )
    selected_bcd_notn: Optional[str] = Field(None, example="005/2017")
    selected_bcd_slno: Optional[str] = Field(None, example="1")

    @field_validator("assessable_value", "quantity", mode="before")
    @classmethod
    def empty_string_to_default(cls, v, info):
        """Converts empty strings or invalid input to default safe values."""
        if v in ("", None):
            return 100000 if info.field_name == "assessable_value" else 100
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return 100000 if info.field_name == "assessable_value" else 100

def get_country_list():
    country_url = "https://www.old.icegate.gov.in/Webappl/Trade-Guide-on-Imports"
    try:
        country_response = requests.get(country_url, headers=headers, timeout=30)
        country_response.raise_for_status()
        country_soup = BeautifulSoup(country_response.text, "html.parser")
        all_countries = country_soup.find("select", {"name": "cntrycd"})

        if not all_countries:
            return []

        countries = []
        for option in all_countries.find_all("option"):
            value = option.get("value")
            if value and "," in value:
                countries.append(value.strip().upper())

        return countries
    except Exception as e:
        print("Error fetching country list:", e)
        return []
    
def resolve_country(input_country: str | None, country_list: list[str]) -> str:
    default = "CN,CHINA"

    if not input_country:
        return default

    normalized = input_country.strip().upper()

    for country in country_list:
        code, name = map(str.strip, country.split(",", 1))
        if normalized in (code, name, country):
            return country

    return default

def fetch_tariff_data(cth_code: str, country: str) -> List[Dict[str, Any]]:
    tariff_url = f"{base_icegate_url}/Webappl/Desc_details"
    tariff_params = {"cth": cth_code, "item_desc": "", "cntrycd": country}

    try:
        tariff_response = requests.get(tariff_url, headers=headers, params=tariff_params, timeout=30)
        tariff_response.raise_for_status()
        tariff_data = tariff_response.json().get("rsAllCth", [])

        return [
            {
                "Tariff Item": item.get("itc_code", ""),
                "Description of Goods": item.get("itc_desc", ""),
                "Unit": item.get("uqc", ""),
                "Rate of Duty": item.get("rta", ""),
                "Import Policy": item.get("itchs_policy", ""),
            }
            for item in tariff_data
            if item.get("itc_code") == cth_code
        ]
    except Exception as e:
        print(f"Error fetching HSN {cth_code}: {e}")
        return []

def fetch_ccr_data(cth_code: str, country: str) -> List[Dict[str, Any]]:
    ccr_url = f"{base_icegate_url}/Webappl/CDC_Desc"
    ccr_params = {"cth_val": cth_code, "cntrycd": country}
    try:
        ccr_response = requests.get(ccr_url, headers=headers, params=ccr_params, timeout=30)
        ccr_response.raise_for_status()
        return ccr_response.json().get("rs_rmsccr_new", [])
    except Exception as e:
        print(f"Error fetching CCR data for {cth_code}: {e}")
        return []

def fetch_swift_data(cth_code: str, country: str) -> List[Dict[str, Any]]:
    """Retrieve country-wise SWIFT PGA Filing data for a specific HSN code."""
    swift_pga_url = f"{base_icegate_url}/Webappl/CDC_Desc"
    swift_pga_params = {"cth_val": cth_code, "cntrycd": country}

    try:
        swift_pga_response = requests.get(swift_pga_url, headers=headers, params=swift_pga_params, timeout=30)
        swift_pga_response.raise_for_status()
        swift_pga_data = swift_pga_response.json().get("rs_swiftpga_new", [])
        return [
            {   "PGA Code": item.get("agency_cd", ""),
                "PGA_Name": item.get("agency_nm", ""),
                "INFO_Code": item.get("info_type_cd", ""),
                "INFO_Desc": item.get("info_type_desc", ""),
                "QFR_Code": item.get("code", ""),
                "QFR Desc": item.get("info_qfr_desc", ""),
                "REQ": item.get("req", ""),
                "Man Opt": item.get("man_opt", ""),
            }
            for item in swift_pga_data
        ]
    except Exception as e:
        print(f"Error fetching SWIFT data for {cth_code}: {e}")
        return []

def fetch_duefee_payloads(cth_code: str, country: str) -> tuple:
    endpoints = {
        "DueFee1": "/Webappl/DueFee1",
        "DueFee111": "/Webappl/DueFee111",
        "DueFee11": "/Webappl/DueFee11",
    }
    params = {"cth_val": cth_code, "cntrycd": country}
    payloads = {}
    for name, path in endpoints.items():
        try:
            url = f"{base_icegate_url}{path}"
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            resp.raise_for_status()
            payloads[name] = resp.json()
        except Exception as e:
            print(f"Error fetching {name} for {cth_code}: {e}")
            payloads[name] = {}
    return payloads.get("DueFee1", {}), payloads.get("DueFee111", {}), payloads.get("DueFee11", {})

def show_igst_notification_SL_no(data: dict) -> str:
    notn, slno = data.get("igst_notn"), data.get("igst_slno")
    return f"{notn}-{slno}" if notn and slno else notn or slno or ""

def calculate_bcd_rate_with_notification(data: dict, notn: Optional[str], slno: Optional[str], bcd_rate: float):
    effective_rate, display = bcd_rate, ""
    if notn:
        for item in data.get("rs_bcdd", []):
            if item.get("notn") == notn and (slno is None or item.get("slno") == slno):
                rta = item.get("rta")
                if isinstance(rta, (int, float)):
                    effective_rate = rta
                display = f"{item.get('notn')}-{item.get('slno')}" if item.get("slno") else item.get("notn")
                break
    return effective_rate, display

def calculate_total_effective_tariff_rate(data: dict, assessable_value: float, override_bcd_rate: Optional[float] = None) -> float:
    get = data.get
    rates = {
        "bcd": override_bcd_rate if override_bcd_rate is not None else get("bcd_rate", 0),
        "aidc": get("aidc_rate", 0),
        "cess": get("cess_rate", 0),
        "chcess": get("chess_rate", 0),
        "adc": get("adc_rate", 0),
        "sws": get("scd_rate", 0),
        "igst": get("igst_rate", 0),
        "cc": get("gstcess_rate", 0),
    }

    BCD = assessable_value * rates["bcd"] / 100
    AIDC = BCD * rates["aidc"] / 100
    CESS = assessable_value * rates["cess"] / 100
    CHCESS = assessable_value * rates["chcess"] / 100
    EAIDC = assessable_value * rates["adc"] / 100
    SWS = (BCD + AIDC + CESS + CHCESS + EAIDC) * rates["sws"] / 100
    IGST = (assessable_value + BCD + AIDC + CESS + SWS) * rates["igst"] / 100
    CC = assessable_value * rates["cc"] / 100

    total = sum([BCD, AIDC, CESS, CHCESS, EAIDC, SWS, IGST, CC])
    return round(total / assessable_value * 100 if assessable_value else 0, 3)

def calculate_total_tariff_rate(data: dict, assessable_value: float) -> float:
    return calculate_total_effective_tariff_rate(data, assessable_value, data.get("bcd_rate", 0))

def build_json_for_calculated_values(cth_code, assessable_value, quantity, selected_bcd_notn=None, selected_bcd_slno=None, country=None):

    assessable_value = assessable_value or 100000
    quantity = quantity or 1

    # 1) fetch
    rate_of_tariff_data, rate_of_effective_data, notification_sl_no_data = fetch_duefee_payloads(cth_code, country)

    # 2) merge (DueFee111 overrides DueFee1)
    data = {**rate_of_tariff_data, **rate_of_effective_data}

    # 3) base rates
    bcd_tariff_rate = data.get("bcd_rate", 0)
    aidc_rate = data.get("aidc_rate", 0)
    cess_rate = data.get("cess_rate", 0)
    cess_spec = data.get("cess_spc_amts", 0)
    sws_rate  = data.get("scd_rate", 0)
    igst_rate = data.get("igst_rate", 0)
    cc_rate   = data.get("gstcess_rate", 0)

    # 4) optional override for BCD
    bcd_effective_rate, bcd_notif_display = calculate_bcd_rate_with_notification(
        notification_sl_no_data, selected_bcd_notn, selected_bcd_slno, bcd_tariff_rate
    )

    try:
        cess_spec_val = float(cess_spec) if cess_spec else 0
    except (TypeError, ValueError):
        cess_spec_val = 0
    
    # 5) duty amounts (use EFFECTIVE BCD)
    BCD  = assessable_value * bcd_effective_rate / 100
    AIDC = BCD * aidc_rate / 100                       # AIDC on BCD
    #CESS = (cess_spec * quantity) if cess_spec else (assessable_value * cess_rate / 100)
    CESS = (cess_spec_val * quantity) if cess_spec_val else (assessable_value * cess_rate / 100)
    SWS  = (BCD + AIDC + CESS) * sws_rate / 100
    IGST = (assessable_value + BCD + AIDC + CESS + SWS) * igst_rate / 100
    CC   = assessable_value * cc_rate / 100


    # 6) build output rows (like your table schema)
    rows = [
        {
            "Customs Duty": "Basic Customs Duty(BCD)",
            "Rate of Duty (Tariff)%": bcd_tariff_rate,
            "Spec Duty": "",
            "Unit": "",
            "Notification -Slno": bcd_notif_display,
            "Rate of Duty (Effective) %": bcd_effective_rate,
            "Spec Duty.1": "",
            "Unit.1": "",
            "Duty Amount": round(BCD, 2),
        },
        {
            "Customs Duty": "Customs AIDC",
            "Rate of Duty (Tariff)%": aidc_rate,
            "Spec Duty": "",
            "Unit": "",
            "Notification -Slno": "",
            "Rate of Duty (Effective) %": aidc_rate,
            "Spec Duty.1": "",
            "Unit.1": "",
            "Duty Amount": round(AIDC, 2),
        },
        {
            "Customs Duty": "Custom Health CESS(CHCESS)",
            "Rate of Duty (Tariff)%": data.get("chess_rate", 0),
            "Spec Duty": data.get("chess_spc_amts", 0) or "",
            "Unit": "",
            "Notification -Slno": "",
            "Rate of Duty (Effective) %": data.get("chess_rate", 0),
            "Spec Duty.1": data.get("chess_spc_amts", 0) or "",
            "Unit.1": "",
            "Duty Amount": 0.0,
        },
        {
            "Customs Duty": "CESS",
            "Rate of Duty (Tariff)%": cess_rate,
            "Spec Duty": cess_spec or "",
            "Unit": rate_of_tariff_data.get("cess_uqc", "") or rate_of_effective_data.get("cess_uqc", "") or "",
            "Notification -Slno": "",
            "Rate of Duty (Effective) %": cess_rate,
            "Spec Duty.1": cess_spec or "",
            "Unit.1": rate_of_tariff_data.get("cess_uqc", "") or rate_of_effective_data.get("cess_uqc", "") or "",
            "Duty Amount": round(CESS, 2),
        },
        {
            "Customs Duty": "Excise AIDC(EAIDC)",
            "Rate of Duty (Tariff)%": data.get("adc_rate", 0),
            "Spec Duty": data.get("adc_spc_amts", 0) or "",
            "Unit": "",
            "Notification -Slno": "",
            "Rate of Duty (Effective) %": data.get("adc_rate", 0),
            "Spec Duty.1": data.get("adc_spc_amts", 0) or "",
            "Unit.1": "",
            "Duty Amount": 0.0,
        },
        {
            "Customs Duty": "Social Welfare Surcharge(SWC)",
            "Rate of Duty (Tariff)%": sws_rate,
            "Spec Duty": "",
            "Unit": "",
            "Notification -Slno": "",
            "Rate of Duty (Effective) %": sws_rate,
            "Spec Duty.1": "",
            "Unit.1": "",
            "Duty Amount": round(SWS, 2),
        },
        {
            "Customs Duty": "IGST Levy",
            "Rate of Duty (Tariff)%": igst_rate,
            "Spec Duty": "",
            "Unit": "",
            "Notification -Slno": show_igst_notification_SL_no(data),
            "Rate of Duty (Effective) %": igst_rate,
            "Spec Duty.1": "",
            "Unit.1": "",
            "Duty Amount": round(IGST, 2),
        },
        {
            "Customs Duty": "Compensation Cess(CC)",
            "Rate of Duty (Tariff)%": cc_rate,
            "Spec Duty": "",
            "Unit": "",
            "Notification -Slno": "",
            "Rate of Duty (Effective) %": cc_rate,
            "Spec Duty.1": "",
            "Unit.1": "",
            "Duty Amount": round(CC, 2),
        },
    ]

    # 7) totals
    total_duty = round(sum(r["Duty Amount"] for r in rows), 2)
    total_tariff_rate    = calculate_total_tariff_rate(data, assessable_value)
    total_effective_rate = calculate_total_effective_tariff_rate(data, assessable_value, override_bcd_rate=bcd_effective_rate)


    total_row = {
        "Customs Duty": "Total Duty",
        "Rate of Duty (Tariff)%": total_tariff_rate,
        "Spec Duty": "",
        "Unit": "",
        "Notification -Slno": "",
        "Rate of Duty (Effective) %": total_effective_rate,
        "Spec Duty.1": "",
        "Unit.1": "",
        "Duty Amount": total_duty,
    }

    # 8) final JSON/dict
    result = {
        "meta_data": {
            "Structure of Duty for CTH": cth_code,
            "country of Origin": country,
            "assessable_value": assessable_value,
            "quantity": quantity,
            # "selected_bcd_notn": selected_bcd_notn,
            # "selected_bcd_slno": selected_bcd_slno,
        },
        "rows": rows,
        "total_row": total_row,
    }
    return result

@app.post("/tariff", summary="Fetch tariff/CCR data and compute duties", tags=["HSN"])
async def tariff_combined(request: HSNRequest):
    """
    Receives a POST request containing a list of HSN codes, then fetches tariff and CCR data for each code and returns the combined results.
    Args:
        request (HSNRequest): Request payload with a list of HSN codes.
    Returns:
        dict: Mapping from each HSN code to its tariff and CCR data, or error information.
    """
    try:
        country_list = get_country_list()
        resolved_country = resolve_country(request.country, country_list)

        result = {}
        for cth in request.hsn_codes:
            result[cth] = {
                "country": resolved_country,
                "tariff_data": fetch_tariff_data(cth,resolved_country),
                "ccr_data": fetch_ccr_data(cth,resolved_country),
                "swift_pga_filing_data": fetch_swift_data(cth,resolved_country),
                "duties": build_json_for_calculated_values(
                    cth,request.assessable_value, request.quantity,
                    request.selected_bcd_notn, request.selected_bcd_slno,
                    resolved_country
                ),
            }
        return result
    except Exception as e:
        return {"error": f"Unexpected error: {e}"}


