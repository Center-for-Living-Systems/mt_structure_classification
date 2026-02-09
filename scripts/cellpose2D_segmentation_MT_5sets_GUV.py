if __name__ == "__main__":
    root_folder = "/mnt/d/lding/CLS/mousumiLiuDinner/raw_data/Microtubule_GUV-Liu-20250106T211105Z-001"
    output_folder = "/mnt/d/lding/CLS/mousumiLiuDinner/set1to5_processed_results"
    dataset_name = "Microtubule_GUV-Liu-20250106T211105Z-001"

    results = run_mt_guv_background_pipeline(
        root_folder=root_folder,
        output_folder=output_folder,
        dataset_name=dataset_name,
        target_shape=(512, 512),
        disk_radius=5,
        save_intermediates=True,
    )

    print("Saved to:", results["processed_folder"])
    print("Pairs:", len(results["df"]))
