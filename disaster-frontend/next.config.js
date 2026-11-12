/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  // Allow API calls to FastAPI backend from server-side
  async rewrites() {
    return [
      {
        source: '/api/backend/:path*',
        destination: `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/:path*`,
      },
    ]
  },
  // Disable image optimization warnings for external URLs
  images: {
    unoptimized: true,
  },
}

module.exports = nextConfig
