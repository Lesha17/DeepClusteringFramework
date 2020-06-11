from clustering_tool.metrics import aggregate_metrics

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', default='output/metrics.csv')
    parser.add_argument('--metrics_dir', default='output/metrics')
    parser.add_argument('--num', type=int, default=5, help='Number of autoencoders trained')
    args = parser.parse_args()

    aggregate_metrics(args.output_file, metrics_dir=args.metrics_dir, num_attempts=args.num)