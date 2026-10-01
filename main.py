import config
from data_exploration import load_data
from data_preprocessing import preprocess_data
from feature_selection import select_features
from model_training import train_model, save_model, load_model
from model_evaluation import evaluate_model
import argparse

def main():
    parser = argparse.ArgumentParser(description='Cancer Prediction Pipeline')
    parser.add_argument('--force-retrain', action='store_true', help='Force model retraining even if saved model exists')
    args = parser.parse_args()

    # 1. Load Data
    df = load_data(config.DATA_PATH)

    # 2. Preprocess
    x_train_norm, x_test_norm, y_train, y_test, labels, classes = preprocess_data(
        df, config.TEST_SIZE, config.RANDOM_STATE
    )

    # 3. Feature Selection
    x_train_selected, x_test_selected = select_features(
        x_train_norm, x_test_norm, y_train, config.N_FEATURES
    )

    # 4. Train or Load Model
    model = load_model(config.MODEL_SAVE_PATH)
    
    if model is None or args.force_retrain:
        model = train_model(x_train_selected, y_train, config.RF_MAX_FEATURES)
        save_model(model, config.MODEL_SAVE_PATH)

    # 5. Evaluate
    evaluate_model(model, x_test_selected, y_test, labels, classes)

if __name__ == '__main__':
    main()
