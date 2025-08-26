import numpy as np
import pandas as pd
import pickle
import xgboost as xgb

def medal_predict_data(request):
    # Original input feature names (before encoding)

    columns = ['front_type', 'front_no_of_colors', 'front_personalisation',
                'back_type', 'back_no_of_colors', 'back_personalisation', 
                'medal_width', 'medal_height', 'medal_thickness', 
                'medal_material','finish', 'second_finish',
                'ribbon_needed', 'ribbon_no_of_colors', 'ribbon_print',
                'no_of_ribbon_print_side', 'ribbon_width', 'ribbon_height', 
                'packaging','quantity']

    # Prepare the input data
    input = [[
        request.front_type, request.front_no_of_colors, request.front_personalisation,
        request.back_type, request.back_no_of_colors,request.back_personalisation,
        request.medal_width, request.medal_height, request.medal_thickness, 
        request.medal_material, request.finish, request.second_finish, 
        request.ribbon_needed, request.ribbon_no_of_colors, request.ribbon_print, 
        request.no_of_ribbon_print_side, request.ribbon_width, request.ribbon_height, 
        request.packaging, request.quantity
    ]]

    predictor = pd.DataFrame(input, columns=columns)

    # OneHotEncoder
    with open('medal_encoders/OneHotEncoder.pkl', 'rb') as file:
        ohe = pickle.load(file)

    # ohe_cols = ['front_personalisation', 'back_personalisation', 'finish', 'ribbon_print',
    #             'second_finish', 'packaging', ]

    ohe_cols = ['front_personalisation', 'back_personalisation', 'finish', 'ribbon_print',
            'second_finish', 'packaging', 'front_type', 'front_no_of_colors', 'back_type', 
        'back_no_of_colors', 'medal_material', 'ribbon_needed', 
        'ribbon_no_of_colors','no_of_ribbon_print_side']

    x_ohe = ohe.transform(predictor[ohe_cols])
    x_ohe = pd.DataFrame(x_ohe, columns=ohe.get_feature_names_out(ohe_cols), index=predictor.index)

    predictor = pd.concat([predictor.drop(columns=ohe_cols), x_ohe], axis=1)

    # LabelEncoder
    # with open('medal_encoders/LabelEncoder.pkl', 'rb') as file:
    #     label_encoders = pickle.load(file)

    # for col, encoder in label_encoders.items():
    #     predictor[col + '_le_encoded'] = encoder.transform(predictor[col])
    #     predictor.drop(columns=[col], inplace=True)

    
    
    # Apply binning + label encoding (same as training)
    def create_bins_and_encode(col, start, end, step):
        bins = list(range(start, end + step, step))
        labels = [f"{i}-{i+step}" for i in range(start, end, step)]
        binned = pd.cut(col, bins=bins, labels=labels, right=True, include_lowest=True)
        
        # Load the same LabelEncoder used in training
        with open('medal_encoders/LabelEncoder.pkl', 'rb') as file:
            label_encoders = pickle.load(file)
        
        col_name = f"bin_{col.name}"
        encoder = label_encoders[col_name]   # must exist from training
        return encoder.transform(binned.astype(str))  # ensure string for consistency

    predictor["bin_medal_width_le_encoded"]   = create_bins_and_encode(predictor["medal_width"], 50, 100, 5)
    predictor["bin_medal_height_le_encoded"]  = create_bins_and_encode(predictor["medal_height"], 50, 100, 5)
    predictor["bin_medal_thickness_le_encoded"]  = create_bins_and_encode(predictor["medal_thickness"], 0, 20, 2)
    predictor["bin_ribbon_width_le_encoded"]  = create_bins_and_encode(predictor["ribbon_width"], 0, 100, 5)
    predictor["bin_ribbon_height_le_encoded"] = create_bins_and_encode(predictor["ribbon_height"], 0, 900, 5)

    # StandardScaler 
    # with open('medal_encoders/StandardScalar.pkl', 'rb') as file:
    #     scalers = pickle.load(file)

    # for col, scaler in scalers.items():
    #     predictor[col + '_scaled'] = scaler.transform(predictor[[col]]).flatten()
    #     predictor.drop(columns=[col], inplace=True)

    # model prediction
    model = xgb.XGBRegressor()
    model.load_model('medal_xgb_model.json')
    output = model.predict(predictor)

    # with open('gradient_boost_model.pkl', 'rb') as file:
    #     model = pickle.load(file)
    # output = model.predict(predictor)



    total_costs = [round(pred, 2) for pred in output.tolist()]

    # adjusted_total_cost = round(total_costs[0] * 1.10, 2)

    quantity = request.quantity

    # Handle single input — avoid divide-by-zero
    cost_per_piece = round(total_costs[0] / quantity, 2) if quantity != 0 else 0.0
    # cost_per_piece = round(adjusted_total_cost/ quantity, 2) if quantity != 0 else 0.0

    response = {
        "total_cost": total_costs[0],
        # "total_cost": adjusted_total_cost,
        "cost_per_piece": cost_per_piece
        }

    
    print(request)

    return response
