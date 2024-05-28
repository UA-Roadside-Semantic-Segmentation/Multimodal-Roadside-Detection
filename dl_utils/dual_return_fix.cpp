int main(int argc, char** argv)
{
    namespace fs = std::filesystem;

    for (const auto & entry : fs::directory_iterator(argv[1]))
    {
        if (fs::is_regular_file(entry) && entry.path().extension() == ".pcd")
        {
            pcl::PointCloud<pcl::PointXYZI> cloud;
            pcl::PCDReader r;
            r.read(entry.path(), cloud);
            if (cloud.width > 4000)
            {
                pcl::PointCloud<pcl::PointXYZI> lastCloud;
                pcl::PointCloud<pcl::PointXYZI> strongestCloud;
                for (int blockPair = 0; blockPair < cloud.size()/4/16; blockPair++) // cloud size guaranteed to be multiple of 12*2*16
                {
                    int blockStart = blockPair*4*16;
                    for (int i = 0; i < 32; i++)
                        lastCloud.push_back(cloud[blockStart + i]);
                    for (int i = 32; i < 64; i++)
                        strongestCloud.push_back(cloud[blockStart + i]);
                }
                cloud.clear();
                for (int i = 0; i < lastCloud.size(); i++)
                {
                    if (lastCloud[i].intensity > strongestCloud[i].intensity)
                        cloud.push_back(lastCloud[i]);
                    else
                        cloud.push_back(strongestCloud[i]);
                }
                cloud.height = 16;
                cloud.width = cloud.size() / 16;
//                std::cout << cloud.width << " " << cloud.height << std::endl;
                std::cout << "Saving " << entry.path().filename() << "..." << std::endl;
                pcl::io::savePCDFileASCII(entry.path().filename(), cloud);
            }
            else
            {
                std::cout << entry.path().filename() << " already less than 4000" << std::endl;
            }
        }
    }
    return 0;
}
