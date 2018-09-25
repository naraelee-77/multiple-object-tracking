function video_path = get_benchmark_path(bench_name)

switch bench_name
    case 'vot13'
        video_path = 'dataset/vot2013';
    case 'vot14'
        video_path = 'dataset/vot2014';
    case 'vot15'
        video_path = 'dataset/vot2015';
    case 'vot16'
        video_path = 'dataset/vot2016'; % ~/ADNet/
    case 'vot17'
        video_path = 'dataset/vot2017'; % ~/ADNet/
end
