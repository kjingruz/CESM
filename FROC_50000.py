import argparse
import os
from coco_froc_analysis.froc import generate_froc_curve, generate_bootstrap_froc_curves

def main():
    parser = argparse.ArgumentParser(description='Run COCO FROC analysis')
    parser.add_argument('--gt_ann', default='output/test_annotations.json', help='Path to ground truth annotations')
    parser.add_argument('--pr_ann', default='50000/predictions_coco.json', help='Path to prediction annotations')
    parser.add_argument('--bootstrap', type=int, default=0, help='Number of bootstrap samples (0 for single run)')
    parser.add_argument('--use_iou', action='store_true', help='Use IoU for matching')
    parser.add_argument('--iou_thres', type=float, default=0.5, help='IoU threshold')
    parser.add_argument('--n_sample_points', type=int, default=100, help='Number of points to sample for FROC curve')
    parser.add_argument('--plot_title', default='FROC Analysis (ResNeXt-101) 50000 iterations', help='Title for the plot')
    parser.add_argument('--plot_output_path', default='50000/froc_analysis.png', help='Output path for the plot')
    
    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(args.plot_output_path), exist_ok=True)

    if args.bootstrap > 0:
        generate_bootstrap_froc_curves(
            gt_ann=args.gt_ann,
            pr_ann=args.pr_ann,
            n_bootstrap_samples=args.bootstrap,
            use_iou=args.use_iou,
            iou_thres=args.iou_thres,
            n_sample_points=args.n_sample_points,
            plot_title=args.plot_title,
            plot_output_path=args.plot_output_path
        )
    else:
        generate_froc_curve(
            gt_ann=args.gt_ann,
            pr_ann=args.pr_ann,
            use_iou=args.use_iou,
            iou_thres=args.iou_thres,
            n_sample_points=args.n_sample_points,
            plot_title=args.plot_title,
            plot_output_path=args.plot_output_path
        )

    print(f"FROC analysis complete. Plot saved to {args.plot_output_path}")

if __name__ == "__main__":
    main()