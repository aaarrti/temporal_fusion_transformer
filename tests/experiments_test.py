from temporal_fusion_transformer.experiments import ElectricityExperiment


def main():
    ElectricityExperiment.from_raw_csv(
        "/Users/artemsereda/Documents/IdeaProjects/temporal_fusion_transformer/data/electricity/LD2011_2014.txt"
    )


if __name__ == "__main__":
    main()
