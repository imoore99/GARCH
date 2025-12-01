import streamlit as st
import os

st.title("🔧 GARCH App Diagnostic")
st.write("✅ App is loading successfully!")

# Test environment variables
st.subheader("Environment Variables")
aws_key = os.environ.get("AWS_ACCESS_KEY_ID", "NOT_FOUND")
aws_secret = os.environ.get("AWS_SECRET_ACCESS_KEY", "NOT_FOUND")

st.write(f"AWS_ACCESS_KEY_ID: {'✅ Found' if aws_key != 'NOT_FOUND' else '❌ Missing'}")
st.write(f"AWS_SECRET_ACCESS_KEY: {'✅ Found' if aws_secret != 'NOT_FOUND' else '❌ Missing'}")

# Test imports
st.subheader("Package Imports")
packages = ['pandas', 'numpy', 'boto3', 'arch', 'yfinance']

for package in packages:
    try:
        __import__(package)
        st.write(f"✅ {package}")
    except ImportError as e:
        st.write(f"❌ {package}: {e}")

st.write("🎉 Diagnostic complete!")