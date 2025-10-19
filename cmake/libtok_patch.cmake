# ------------------------------------------------------------------------
# Temporary patches for LibTok
# ------------------------------------------------------------------------

set(LIBTOK_SOURCE_DIR "${LIBTOK_ROOTDIR}/source")

# ------------------------------------------------------------------------
# 1 - Fix encodeTokenIds recursive call
# ------------------------------------------------------------------------
set(LIBTOK_FILE_TO_PATCH "${LIBTOK_SOURCE_DIR}/token_id_converter.cpp")

if(EXISTS "${LIBTOK_FILE_TO_PATCH}")
    message(STATUS "Patching LibTok bug in ${LIBTOK_FILE_TO_PATCH}")

    file(READ "${LIBTOK_FILE_TO_PATCH}" file_content)

    # Replace the buggy recursive call with the qualified one
    string(REPLACE
        "encodeTokenIds(tokSeq);"
        "this->TokenIdConverterBase::encodeTokenIds(tokSeq); // [TEMP PATCH APPLIED]"
        file_content "${file_content}"
    )

    file(WRITE "${LIBTOK_FILE_TO_PATCH}" "${file_content}")
    message(STATUS "→ Applied temporary fix to TokenIdConverterBase::encodeTokenIds")
else()
    message(WARNING "LibTok patch skipped — file not found: ${LIBTOK_FILE_TO_PATCH}")
endif()


# ------------------------------------------------------------------------
# 2 - Fix TokSequence::sliceUntilEnd out-of-range crash
# ------------------------------------------------------------------------
set(LIBTOK_SLICE_FILE "${LIBTOK_SOURCE_DIR}/tok_sequence.cpp")

if(EXISTS "${LIBTOK_SLICE_FILE}")
    message(STATUS "Patching LibTok sliceUntilEnd safety fix in ${LIBTOK_SLICE_FILE}")

    file(READ "${LIBTOK_SLICE_FILE}" slice_content)

    # Match and replace the old implementation with the safe version
    string(REGEX REPLACE
        "TokSequence TokSequence::sliceUntilEnd\\([^)]*\\) const[^{]*\\{[^}]*\\}"
        "TokSequence TokSequence::sliceUntilEnd(size_t start) const\n{\n    TokSequence seq = *this;\n\n    auto safe_slice = [&](auto& out, const auto& in) {\n        if (start < in.size())\n            out = typename std::decay_t<decltype(out)>(in.begin() + start, in.end());\n        else\n            out.clear();\n    };\n\n    safe_slice(seq.tokens, tokens);\n    safe_slice(seq.ids, ids);\n    safe_slice(seq.events, events);\n\n    if (start < bytes.size())\n        seq.bytes = bytes.substr(start);\n    else\n        seq.bytes.clear();\n\n    return seq;\n}"
        slice_content "${slice_content}"
    )

    file(WRITE "${LIBTOK_SLICE_FILE}" "${slice_content}")
    message(STATUS "→ Applied safe TokSequence::sliceUntilEnd() fix")

else()
    message(WARNING "LibTok sliceUntilEnd patch skipped — file not found: ${LIBTOK_SLICE_FILE}")
endif()
