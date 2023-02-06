/**
 * @type {import('next').NextConfig}
 */
const nextConfig = {
    basePath: process.env.BASE_PATH,
    assetPrefix: process.env.BASE_PATH + '/'    
}

module.exports = nextConfig
