import pandas as pd
import pickle


def predict_data(request):
    # Original input feature names (before encoding)

    columns = ['front_type', 'front_no_of_colors', 'front_personalisation',
                'back_type', 'back_no_of_colors', 'back_personalisation', 
                'medal_width', 'medal_height', 'medal_thickness', 
                'finish', 'second_finish','double_finish', 
                'ribbon_needed', 'ribbon_no_of_colors', 'ribbon_print',
                'no_of_ribbon_print_side', 'ribbon_width', 'ribbon_height', 
                'packaging','attachment','quantity']

    # Prepare the input data
    input = [[
        request.front_type, request.front_no_of_colors, request.front_personalisation,
        request.back_type, request.back_no_of_colors,request.back_personalisation,
        request.medal_width, request.medal_height, request.medal_thickness, 
        request.finish, request.second_finish, request.double_finish, 
        request.ribbon_needed, request.ribbon_no_of_colors, request.ribbon_print, 
        request.no_of_ribbon_print_side, request.ribbon_width, request.ribbon_height, 
        request.packaging, request.attachment, request.quantity
    ]]

    predictor = pd.DataFrame(input, columns=columns)

    # OneHotEncoder
    with open('encoders/OneHotEncoder.pkl', 'rb') as file:
        ohe = pickle.load(file)

    ohe_cols = ['front_personalisation', 'back_personalisation', 'finish', 'ribbon_print',
                'second_finish', 'packaging', 'attachment']

    x_ohe = ohe.transform(predictor[ohe_cols])
    x_ohe = pd.DataFrame(x_ohe, columns=ohe.get_feature_names_out(ohe_cols), index=predictor.index)

    predictor = pd.concat([predictor.drop(columns=ohe_cols), x_ohe], axis=1)

    # LabelEncoder
    with open('encoders/LabelEncoder.pkl', 'rb') as file:
        label_encoders = pickle.load(file)

    for col, encoder in label_encoders.items():
        predictor[col + '_le_encoded'] = encoder.transform(predictor[col])
        predictor.drop(columns=[col], inplace=True)

    # StandardScaler 
    with open('encoders/StandardScalar.pkl', 'rb') as file:
        scalers = pickle.load(file)

    for col, scaler in scalers.items():
        predictor[col + '_scaled'] = scaler.transform(predictor[[col]]).flatten()
        predictor.drop(columns=[col], inplace=True)

    # Model Prediction (Gradient Boosting) 
    # model = joblib.load('gradient_boosting_model.pkl')
    # output = model.predict(predictor)

    with open('gradient_boosting_model.pkl', 'rb') as file:
        model = pickle.load(file)
    output = model.predict(predictor)

    response = {
        # "result": output.tolist()
        "result": [round(pred, 2) for pred in output.tolist()]
        }

    return response
