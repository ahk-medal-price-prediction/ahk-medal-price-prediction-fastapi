from pydantic import BaseModel

class Data(BaseModel):
    front_type: str
    front_no_of_colors: int
    front_personalisation: str
    back_type: str
    back_no_of_colors: int
    back_personalisation: str
    medal_width: int
    medal_height: int
    medal_thickness: int
    finish: str
    second_finish: str
    double_finish: str
    ribbon_needed: str
    ribbon_no_of_colors: int
    ribbon_print: str
    no_of_ribbon_print_side: int
    ribbon_width: int
    ribbon_height: int
    packaging: str
    attachment: str
    quantity: int
